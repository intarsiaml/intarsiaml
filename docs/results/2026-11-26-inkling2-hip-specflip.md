# Inkling-2-Mini HIP spec default flips (lane ink-hip-specflip)

Status: in progress (2026-11-26 22:55 EST). Flip B: confirmation ABBA **meets GO** again; KV check 2 **over 5 %** by the pre-registered
reading (memory-limited solo capacity -9.9 %; the default contexts do not move): flip B stopped, (a) / (b) / (c) go to the user. Flip A: PR A.

## Speed table: confirmation ABBA, DFlash2 k4 / DSpark k5 (windows C1 + C2, card 1, binary `main7`)

`main7` = origin/main `cd16cac1b` (#541 merged), md5 `689cde21f795f1e38ba968c25d06daea`: the same bits as #541's ABBA binary `dfl6` (the
code is identical; only docs differ). 4 + 4 fresh processes (C1 A B B A r1 r2, C2 B A A B r3 r4), card 1 (#541: card 0), #541's
prompts / arms / `speed.py`. Ratio per (prompt, rep), geometric mean per cell, 95 % CI = t over the 4 rep pairs (`raw/conf-k4-analysis.txt`).

| DFlash2 k4 / DSpark k5 | short | 1k | prose | chat | 4k | 16k | **mean of the 6 (geo / arith)** | code (extra) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| confirmation, card 1 | **1.190** [1.184-1.196] | **1.142** [1.137-1.147] | **1.047** [1.044-1.051] | **1.217** [1.211-1.223] | **0.991** [0.987-0.994] | **1.101** [1.097-1.105] | **1.112 / 1.115** | 1.303 [1.299-1.308] |
| #541, card 0 | 1.189 [1.184-1.194] | 1.142 [1.138-1.146] | 1.048 [1.046-1.050] | 1.216 [1.212-1.221] | 0.991 [0.987-0.995] | 1.101 [1.098-1.104] | 1.112 / 1.115 | 1.303 [1.299-1.306] |

| tok/s (median of 4 processes) | short | 1k | prose | chat | 4k | 16k | code |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSpark k5 (main's default) | 109.83 | 102.24 | 121.39 | 100.63 | 117.42 | 91.23 | 127.71 |
| DFlash2 k4 | 130.77 | 116.89 | 127.23 | 122.54 | 116.45 | 100.53 | 166.54 |

Prefill per class DFlash2 / DSpark 0.98-1.00 (4k 3443 vs 3446, 16k 3252 vs 3260 tok/s). Tokens per verify pass 3.32 vs 2.72 (pooled).

**GO verdict: met again.** Geometric mean 1.112 >= 1.03; worst cell 4k 0.991 with its CI upper bound 0.994, not below 0.99 (as narrow as
#541's 0.995); ids identical across all 8 processes, and equal to #541's card-0 ids (md5 of the `ids[...]` lines `4f5a5e398f69...`, both arms).
The two cards agree to the third decimal on every cell.

## KV check (coordinator check 2): holding both drafters

Card 1, binary `main7`, solo `generate` (`raw/kv/`): with `--max-seq 1000000` the context printed is the memory-limited KV capacity
(`memplan.rs`: `0.5 x measured free memory / bytes per token`, the default `--kv-fraction`); the defaults (`--max-seq` 4096) after.

| load | weights line | KV capacity (tokens, `--max-seq 1000000`) | default context |
|---|---:|---:|---:|
| `--spec off` | 14.97 GB | 192,960 | |
| DSpark alone (main's default) | 15.94 GB | 177,504 | 4096 (`raw/kv/dspark-default.err`) |
| DFlash2 alone (`--spec dflash --spec-draft <DFlash2> --spec-k 4`) | 16.05 GB | 175,424 | |
| DSpark + DFlash2 (cannot load today; derived) | 17.02 GB | **159,968** | 4096 (159,968 > 4096) |
| `serve --max-batch 16`, any of them | | | no drafter loads: `--spec dflash` is refused ("--spec dflash with the batching engine (use --max-batch 1)", `raw/kv/b16-dfl.err`) and the DSpark default is off above one sequence; the batch-16 context (#538: 2791 per sequence) is the same whatever drafter flips. The b16 server did not come up inside the 55 s the stage gave it (first-start admission), so its line comes from the gate window. |

Derivation (the engine's own formula, applied by the loads themselves): DFlash2's resident bytes (weights, its mxfp4 head copy, tap
buffers) take capacity(off) - capacity(DFlash2) = 192,960 - 175,424 = **17,536 tokens**; DSpark's take 192,960 - 177,504 = 15,456. Both
together: 177,504 - 17,536 = **159,968**, i.e. **-9.9 %** of DSpark-alone capacity (17,536 / 177,504). About 32 KB of KV budget per
token (DSpark: ~1.0 GB of resident bytes, half of it out of the KV budget, over 15,456 tokens). The draft per-sequence state (DFlash2:
5 layers x 8 x 128 x 2 x 2 B = 20 KB per token of cap, full length) is allocated after the plan, outside the budget.

**Verdict, as pre-registered: over 5 %** (the memory-limited solo capacity, -9.9 %). The default contexts do not move: solo stays 4096
(the capacity stays 39x above it) and `--max-batch 16` loads no drafter. So the cost is real only for a solo server run with a large
`--max-seq`. Flip B stops here; the choice among (a) dual drafters (-9.9 % solo capacity), (b) DFlash2 alone, sampled requests drafting
with DFlash2 (capacity -1.2 % vs main; sampled DFlash2 is unmeasured and needs its own gate) and (c) DFlash2 alone, sampled requests
plain on HIP (flip A lost wherever B applies) goes to the user.

## Flip A as built (PR A)

- **Rule.** A new recipe key `[tuning.<backend>] sampled_dflash` (bool; `recipes/inkling2.toml` sets `true` under `[tuning.hip]` only).
  `Model::spec_sampled_dflash` resolves `Model::set_spec_sampled_dflash` (tests), then `INTARSIA_SPEC_SAMPLED_DFLASH`, then the recipe
  key, then off (`spec::sampled_dflash_resolve`). The key counts only when the loaded draft is the checkpoint's own DSpark draft: an
  explicit `--spec dflash --spec-draft <DFlash2 dir>` keeps sampled requests plain unless the variable says `1` (sampled DFlash2 was
  never measured). Where it is on, #543's draft length policy (`INTARSIA_SPEC_SAMPLED_ADAPT`, recipe `[tuning.hip] sampled_adapt`) is
  on with it, as #543 built it.
- **Why a recipe key, not `spec::sampled_dflash_default_for("hip")`:** a code default per backend would also turn sampled drafting on
  for every other model that drafts with a DFlash-family draft on HIP (Qwen3.8-27B DFlash2 / DSpine run there); the recipe scopes it to
  Inkling-2, like `gumbel_sampling` on Metal.
- **Unchanged:** Metal (it failed #543's gate), CUDA, cpu; greedy requests; `--spec off`; `--max-batch` > 1 (the batching engine loads
  no draft at all, so it takes plain steps as on main); prefix-store requests (the DSpark default is off with a store, as on main).
- **Revert:** `INTARSIA_SPEC_SAMPLED_DFLASH=0` (or delete the recipe line). `INTARSIA_SPEC_SAMPLED_ADAPT=0` keeps sampled drafting with
  fixed k5 (#535's form).

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

**Lane to coordinator, ~22:55 EST (lane clock):** confirmation ABBA meets GO again; KV check 2 over 5 % (-9.9 % solo capacity with both
drafters; default contexts unchanged): flip B stopped, (a) / (b) / (c) to the user.

**Coordinator, sent ~23:10 EST 2026-11-26 (the message said "~23:55"; corrected by the coordinator, see below), as written:**

> **USER DECISION (23:55 [corrected: ~23:10 EST 11-26], verbatim): "Test DFlash2 sampled", option (b).** Under flip B, when the DFlash2 drafter is present, sampled requests draft with DFlash2 too. That leaves one drafter loaded, at −1.2 % capacity vs main. When DFlash2 is absent, the server behaves exactly like flip A (sampled DSpark adaptive). Option (a), two drafters, is not built now.
>
> **Order:**
> 1. PR A (flip A) continues now, on its own. #543 merged as 3fb1b5f8d.
> 2. **The sampled DFlash2 test.** Reuse #543's harness (`docs/results/2026-11-26-inkling2-sampled-adapt.md`, its `raw/` and `sim.py`).
>    - Run DFlash2 k4 with the adaptive policy on HIP.
>    - Calibrate the adapt table for DFlash2 from one trace window with `sim.py`. DSpark's table is likely wrong for DFlash2's acceptance.
>    - Then take one measurement window: paired on/off (plain sampling), T 1.0 and T 0.7, prompt classes short / 1k / 4k plus `short_b`, n ≥ 3 reps.
> 3. **Gate, pre-registered. Do not relax it:**
>    1. Every prompt is ≥ 0.98 of plain sampling, `short_b` included, at both temperatures.
>    2. Each class's on/off ratio is ≥ 0.98 × the DSpark-adaptive ratio from #543 on HIP:
>       - T 1.0: short 1.172, 1k 1.042, 4k 1.466;
>       - T 0.7: short 1.163, 1k 1.068, 4k 1.458.
>       Without this, (b) could make sampled requests slower than flip A.
>    3. 100 seeds at T 1.0 produce ids equal to `--spec off`.
>    4. Greedy output is unchanged from your confirmed DFlash2 k4.
> 4. **If the gate passes,** PR B is: DFlash2 k4 for greedy and for sampled (adaptive, with the DFlash2 table) when DFlash2 is found; otherwise DSpark as today.
>    - Revert envs: one restores DSpark k5 for greedy, and A's switch still turns sampled drafting off.
>    - Its gates are the ones in your brief, plus a fallback test with no drafter present.
> 5. **If the gate fails,** go to (a), two drafters, per the plan the user saw. Before writing code, report the numbers to me in ≤ 6 lines. I will relay them and give the go.
>
> **Radiance:** a new 10 min for steps 2-3, on top of the 6:10 you have left, which stays reserved for the PR A / PR B gate window. That is 16:10 in all. Each window ≤ 5 min, none after 06:00. Metal stays at ≤ 15 min.

**Lane to coordinator, ~23:14 EST:** PR A gate window G1 (#12441, 3:35): greedy md5 default = `--spec off` (380af8a5), 100 / 100 seeds
identical (off 0 passes, default 265, startup line "drafting on (recipe ...)"), contexts solo 4096 / b16 3285; serve 16/16 did not run
(client URL lacked `/v1`; the solo first-start admission passed 150 s beside the b16 one). Asked for G2 (serve-only) and a PR B gate window.

**Coordinator, sent 23:15 EST 2026-11-26 (the message said "2026-11-27 ~00:20"; corrected by the coordinator), as written:**

> **Yes to both:**
> - G2, serve-only, ≤ 5:00 of radiance now.
> - PR B's gate window, ≤ 5:00 later.
>
> Your lane's total cap tonight is **33:00**. You have used 12:25. The 10 min for the DFlash2 sampled test (D1 calibration and D2 measurement) are inside that cap. Each window ≤ 5 min, none after 06:00.
>
> **For G2:**
> 1. Fix the client URL so it includes `/v1`, and check it against radiance's host format before the window.
> 2. The serve check is 16/16 replies at `--max-batch 16` equal to solo.
> 3. Also record main 3fb1b5f8d's b16 context line, from an existing log if there is one, so 3285 is shown to be unchanged by flip A.
>
> **To cut the cost of PR B's gate:**
> - Build the PR B candidate binary before D2 and run D2's measurement with it. Its admission cache is then warm for the gate window.
> - If D1 or D2 has a spare card while the other card measures, fold gate checks into it.
> - Never let a gate check share a card with a timed speed run.
>
> **If G2 fails for a real reason** (replies differ, not a harness error), stop PR A and report to me.

**Coordinator correction (as written):** two timestamps in my last two messages were wrong. The user's decision ("Test DFlash2 sampled")
came at about 23:10 EST on 2026-11-26, not 23:55. My G2 budget grant was sent at 23:15 EST on 11-26, not 00:20 on 11-27. Correct both in
your results doc. Nothing else changes. The 06:00 cutoff for windows is 06:00 EST on 11-27.
