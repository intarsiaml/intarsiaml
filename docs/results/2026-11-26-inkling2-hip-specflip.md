# Inkling-2-Mini HIP spec default flips (lane ink-hip-specflip)

Status: in progress (2026-11-26 22:25 EST).

## Speed table

(Confirmation ABBA for flip B: pending.)

## Pre-registration: confirmation ABBA and KV check (written 22:40 EST by the lane clock, before any window)

**Binary.** `main7` = the exact tree of origin/main `cd16cac1b` (#541 merged; `git archive`, nothing of this lane), built on tatooine in
the lane volume (seeded from ink-dflash2's). Its md5 goes in the table.

**ABBA.** #541's harness unchanged (`raw/stage_ab_k.sh`, a copy with this lane's paths): `raw/speed.json` (md5 `7763e4744bb4abcfef7a19552708b9cf`),
256 tokens, greedy, EOS honoured, one process at a time, **card 1** (`HIP_VISIBLE_DEVICES=1`; #541 ran card 0). Arms: `dspark` = no
`--spec` flags (the recipe default, DSpark k5, cut 0.5); `dfl` = `--spec dflash --spec-draft /models/Inkling-2-Mini-DFlash2 --spec-k 4`.
Window C1 = A B B A (reps r1 r2), window C2 = B A A B (reps r3 r4): 4 + 4 fresh processes. Analysis `raw/speed.py <dir> dspark
"short=1-12,1k=13,4k=14,16k=15,prose=16-17,chat=5-12,code=3+18"` (#541's class string; code is an extra cell, not in the rule): ratio per
(prompt, rep), geometric mean per cell, 95 % CI = t over the 4 rep pairs.

**GO rule (#541's, unchanged):** the **geometric** mean of the six cells (short, 1k, prose, chat, 4k, 16k) >= 1.03 (the arithmetic mean
is reported too; the geometric one decides, as #541's headline), no cell with its 95 % CI upper bound below 0.99, and ids identical across
all 8 processes. A fail ends flip B (DFlash2 stays opt-in); no other k, no re-run.

**KV check (coordinator check 2), same windows, card 1, after the ABBA processes, `raw/stage_kv.sh`:**
- C1: solo `generate` at `--max-seq 1000000` (the context then shows the memory-limited KV capacity, the engine's own formula
  `0.5 x measured free memory / bytes per token`, `memplan.rs`): `--spec off`, default (DSpark k5), DFlash2 k4.
- C2: the recipe defaults: solo default (DSpark), `serve --max-batch 16` default, `serve --max-batch 16 --spec dflash ...` (expected:
  refused, no drafter loads with batching).
- Both drafters cannot load today, so: capacity(DSpark + DFlash2) = capacity(DSpark) - (capacity(off) - capacity(DFlash2)) (DFlash2's
  resident bytes = the free-memory drop it causes, converted at the same bytes per token). **Decisive number:** that solo capacity loss
  as a share of capacity(DSpark); > 5 % = stop and report (a) / (b) / (c) to the coordinator. Also reported: the default contexts (solo
  `--max-seq` 4096, `--max-batch` 16), which move only if the capacity falls below them.

## Coordinator brief (verbatim summary, 2026-11-26 ~22:25 EST)

The user answered "Both, confirm first" (~22:30 EST). That approves two default
flips, both for **Inkling-2-Mini on HIP (R9700) only**.

**Flip A. Sampled requests: sampled DSpark with adaptive draft length on by default.**
- Today opt-in with `INTARSIA_SPEC_SAMPLED_DFLASH=1` (#535, main `aee06201c`). Adaptive policy in PR #543.
- No new speed gate. #543 passed on HIP: story `short_b` at 0.981 of plain sampling; seeded output equals `--spec off`; greedy equals main.
- Metal failed that gate, so Metal stays opt-in. CUDA and cpu unchanged.

**Flip B. Greedy drafter: DSpark k5 -> DFlash2 k4.**
- Only if a **confirmation ABBA** meets the pre-registered GO rule again: mean of six cells (short, 1k, prose, chat, 4k, 16k) >= 1.03 vs DSpark k5; no cell's 95 % CI upper bound below 0.99.
- First run (PR #541, n=4, tatooine card 1, binary `dfl6`): short 1.189, 1k 1.142, prose 1.048, chat 1.216, 4k 0.991 [0.987-0.995], 16k 1.101, mean 1.112.
- Confirmation: 4 + 4, fresh processes, **the other card** (card 1 = `tatooine-r9700-2`), binary built from main **after #541 merges**. Same prompts, harness, analysis as #541 (`docs/results/2026-11-26-inkling2-dflash2.md`, `raw/stage_ab_k.sh`, `raw/kchoice.py`, `raw/abk4-analysis.txt`).
- If it fails: no flip B; DFlash2 stays opt-in; record numbers and report. No other k values, no re-run hoping for a pass.

**Where things are**
- Worktree `/opt/code/intarsiaml/.worktrees/ink-hip-specflip`, branch `ink/hip-specflip` from origin/main `aee06201c`; scratch under `.scratch/`.
- tatooine: `~/intarsiaml-ink-hip-specflip` and docker volume `intarsia-cargo-target-ink-hip-specflip` are the only new remote folder/volume. May read `~/intarsiaml-ink-dflash2` (drafter `/data/models/intarsia/Inkling-2-Mini-DFlash2`) and `~/intarsiaml-ink-sampled-adapt`.
- Input PRs merge in background, in order #541, #543, #544. Until #541 and #543 are MERGED, only read and plan (poll <= every 4 min). Then merge `origin/main` into the branch (merge, never rebase). If either still open at 00:30, report rather than wait.

**PR A (flip A)**
- On HIP with Inkling-2, a sampled request uses sampled DSpark + adaptive policy by default; Metal, CUDA, cpu unchanged.
- Revert switch: one env value restores today's behaviour (e.g. `INTARSIA_SPEC_SAMPLED_DFLASH=0`), documented in `docs/tuning.md` and `docs/recipes.md`.
- cpu tests: default on for hip, off for metal/cuda/cpu; revert switch works; seeded ids still equal `--spec off`. LOG.md row.

**PR B (flip B, only if confirmation passes)**
- On HIP with Inkling-2, default greedy spec is DFlash2 k4 when the DFlash2 drafter can be found, else DSpark k5. Missing drafter never an error.
- Locate drafter without a flag (likely a recipe field with the path; justify in doc).
- Revert switch: one env value restores DSpark k5.
- Tests: fallback with no drafter; revert switch; `--spec off` unchanged.

**Combined cases**
- Both flips on: sampled requests still draft with DSpark, so the server loads both drafters. Measure the extra drafter weights' effect on default context and KV budget at recipe defaults (`--kv-fraction`, `--max-seq`, and `--max-batch 16`). #538: default budget tight, context clamped to ~2791 tokens. Record before/after; if clamp worsens say by how much; if it costs > 5 % of default KV tokens, stop and ask before merging B.
- `--max-batch` > 1: state and test what the default spec does; must be unchanged from main or bitwise equal to solo.

**Gates (coordinator merges on these)**
1. Full cpu suite on final head (core lib, scheduler, models tests, CLI) on endor via `fleetq run --host endor --load cpu`.
2. One HIP window on merged-main binary: greedy md5 for new defaults == `--spec off` (short, 4k, 256 tokens); 100 sampled seeds == `--spec off` at T 1.0; serve 16/16 at `--max-batch 16` == solo; default context and KV log line.
3. Metal no-op check on endor: Metal greedy md5 main vs branch; Metal tiny suite; one `--exclusive` slot <= 15 min.
4. For B: confirmation ABBA passes GO rule.

**Budget**: radiance <= 15 min total (~2 confirmation windows of ~3.75 min + one gate window), each window <= 5 min, none starting after 06:00 EST. Metal: one exclusive slot <= 15 min.

**Rules**: all fleet commands via fleetq with honest `--expect` (builds/tests `--load cpu`; windows `--host tatooine-r9700` [+ `tatooine-r9700-2` for card 1] `--takes-down radiance --quiet gpu,cpu`). Never `fleetq release` a running window job; to abort, kill the driver's process tree then confirm `curl -s http://tatooine:8080/v1/models`. Confirm radiance after every window. No tool call/wait > 4 min. Push via `fleetq run --host endor --load cpu`. PR body ends with the Claude Code line; commits end with Co-Authored-By. Do not merge. Results doc speed table first; capture outcome in unanimis (`unim_capture_new`, no project/scope, `intarsiaml` in source). No new folders beyond those named; delete none.

**Final report (<= 15 lines)**: confirmation table (per-cell ratio + 95 % CI, mean, GO verdict); PR A/B numbers + head shas; each gate's result; KV/context before/after; radiance and Metal minutes; what is not verified; fleetq ids.

## Later go-aheads from the coordinator

**Lane to coordinator, ~22:35 EST (lane clock):** flip B as briefed needs the engine to hold two drafters at once (it holds exactly one:
`LoadOptions.dflash`, `Model.dflash`, `SequenceState.dflash`, one SpecConfig k; the Inkling tap layout comes from the one head).
Options: (a) build dual-drafter support (DSpark primary as on main, DFlash2 a greedy-only add-on head, a parked second draft state
swapped per request, head resolution per state, per-drafter k, the head in the graph key); (b) under B sampled requests draft with
DFlash2 (seed-exact, unmeasured, own gate); (c) under B sampled requests take plain steps (flip A lost on HIP). Asked: Q1 (a) OK?
Q2 build the confirmation binary from main cd16cac1b now?

**Coordinator, 2026-11-26 ~23:05 EST (as written):**

> **Q2: yes.** Build the confirmation-ABBA binary from main cd16cac1b now and run the 2 windows on card 1. Nothing about the budget or rules changes.
>
> **Q1: (a) is OK, but only after two checks pass, in this order:**
> 1. **The confirmation ABBA meets GO.** If it fails, flip B is off and (a) is not needed. Record the result and report.
> 2. **The KV cost of holding both drafters is ≤ 5 % of default KV tokens.** Measure it before you write any (a) code.
>    - Use the card-1 side of the ABBA windows. Start the server at the recipe defaults (`--kv-fraction`, `--max-seq`, and also `--max-batch 16`) three ways:
>      - with DSpark alone (main);
>      - with DFlash2 alone (`--spec dflash`);
>      - DSpark plus DFlash2: if that cannot load today, add DFlash2's resident bytes (weights plus workspace) to the DSpark run and convert them to KV tokens with the engine's own budget formula, and show the arithmetic.
>    - Record the context and KV log lines for each.
>    - My rough estimate: about 945 MB extra at roughly 0.2 MiB per KV token is about 2.3k-4.7k tokens out of 44,656, or about 5-10 %. So this may well fail.
>    - If it is over 5 %, stop and report with the numbers and the three options (a), (b) and (c). That choice goes to the user.
>
> **PR A (flip A) must not wait for (a).** Once #543 merges, build and ship PR A on its own, with its own gates.
>
> **If (a) goes ahead:**
> - Keep it in PR B, separate from PR A.
> - The engine test must cover:
>   - a missing drafter or the revert env gives a byte-identical load to main (same head count and same KV line);
>   - a greedy turn followed by a sampled turn in one conversation keeps the ids equal to `--spec off`.
> - The Metal no-op gate applies to PR B as well.
