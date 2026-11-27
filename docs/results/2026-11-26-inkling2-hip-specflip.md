# Inkling-2-Mini HIP spec default flips (lane ink-hip-specflip)

Status (2026-11-27 00:33 EST): **flip A = PR #545**: on the final head `136b4dc16` the cpu suite (#12480) and the HIP gates (GA2 md5 /
seeds / contexts, GS2 serve 16 / 16) pass; the Metal slot is pending (endor queue). **Flip B (b)** (DFlash2 alone, sampled requests drafting with it)
**failed its pre-registered gate** (D2). **Coordinator 00:08: build (a)** (both drafters: greedy DFlash2 k4, sampled DSpark adaptive);
its gate is pre-registered below ("Pre-registration: the (a) gate"), before any (a) window.

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

## PR A gates (HIP: windows G1, G2, G2b, G2c)

Binary `specA1` / `specA1g` = d84acc9f6 (flip A on main `3fb1b5f8d`), md5 `d52cbb09447871313b6c3261a4de2f75`; later PR A commits are
tests, harness and docs only (the `load.rs` addition is `#[cfg(test)]`). Main's binary `mainA` = `3fb1b5f8d`, md5 `22ae5501...`.

| check (HIP, R9700) | result | where |
|---|---|---|
| greedy md5, md5.json (golden `code` 101 ids + the 4108-id prompt), 256 tokens, `--ignore-eos`: the new default (DSpark k5) vs `--spec off` | **equal**, `380af8a5e897bdbed231818477038f37` both (the same as #541's gate); default 122.8 vs off 81.8 tok/s | G1 card 1, `raw/gate-specA1/md5-*` |
| 100 seeds at T 1.0, 24 tokens, k5 (real-model harness): `INTARSIA_SPEC_SAMPLED_DFLASH=0` vs the default (unset) | **100 / 100 identical**; not vacuous: off 0 verify passes, default 265 passes (408 proposed, 123 accepted); default startup line `DFlash-family drafting on (recipe inkling2 [tuning.hip] sampled_dflash; ...)` and the recipe's draft length policy | G1 card 0, `raw/gate-specA1/seeds-*` |
| serve `--max-batch 16` (16 greedy replies at once) vs serve solo (the default, one at a time), 128 tokens | **16 / 16 equal** (G2c). G1: client URL without `/v1` (404); G2: 400 (two prompts longer than the b16 context); G2b: 15 / 16, the 16th prompt (3247 tokens) hit the b16 context after 38 tokens (its 37 first tokens' text equal to solo's; the 38th decoded as a NUL character: an edge of the batching engine at a full context, flip A does not touch it, not investigated); G2c with that prompt cut to 7600 characters: 16 / 16 | `raw/servec-specA1g/serve-compare.txt` |
| default context / KV lines | solo default: `context 4096 tokens`, weights 15.94 GB (DSpark loaded); `--max-batch 16`: `context 3285 tokens` on the branch and **3285 on main 3fb1b5f8d** (unchanged by flip A: no draft loads with batching) | `raw/gate-specA1/sampled-def.err`, `raw/serve-specA1g/{b16,main-b16}.log` |

Radiance: G1 #12441 3:35, G2 #12454 2:27, G2b #12461 0:58, G2c #12462 0:59.

**GA2 (#12493, 00:15:34-00:19:25, 3:51; binary `specA2` = PR A final head `136b4dc16`, main `1bf40e29f` with #544 merged in, md5
`8e927dd520f2765ad8da92db4f198c35`; `raw/gate-specA2/`, `raw/window-ga2-12493.log`):**
- greedy md5 (md5.json, 256 tokens): default (DSpark k5) = `--spec off` = `380af8a5e897bdbed231818477038f37` (default 122 vs off 81.9 tok/s).
- 100 seeds at T 1.0: `INTARSIA_SPEC_SAMPLED_DFLASH=0` vs default **100 / 100 identical**; not vacuous (off 0 verify passes; default 265
  passes, 408 proposed, 123 accepted).
- contexts: solo default `context 4096 tokens`, weights 15.94 GB; `--max-batch 16`: `context 3285 tokens` (cold admission, 134 cases, 104 s).
- **serve b16 vs solo: not run.** The solo server (card 1) was still in its cold first-start admission (new binary, no cached cases) when
  the stage's 175 s wait ran out; `solo.json` missing. G2c's 16 / 16 was on `specA1g` (before #544). Needs a serve-only rerun (asked).

**GS2 (#12499, 00:28:34-00:31:29, 2:55; serve-only rerun, coordinator 00:27, one try; `specA2`; `raw/serve2-specA2/`,
`raw/window-gs2-12499.log`):** solo (card 1) first and alone: up after 120 s (cold admission), 16 replies one at a time; then b16 (card 0,
admission cached: `0 case(s) run, 134 cached`, up after 1 s) 16 replies at once, `context 3285 tokens`. **b16 vs solo: 16 / 16 replies
equal** (128 tokens each, prompts 2807-2864 tokens for the 4k cuts). Gate 2 on the final head: **pass**.

Radiance so far: C1 + C2 + G1 + G2 + G2b + G2c + D1 + D2 = 21:51 (coordinator's tally), GA2 3:51, GS2 2:55: **28:37** of 33.

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

## D1: sampled DFlash2 calibration (window D1 #12463, 2:24, binary specA1)

Fixed k4 sampled DFlash2 (`INTARSIA_SPEC_SAMPLED_ADAPT=0`), all 6 prompts at 128 tokens: T 1.0 85.5 tok/s (acceptance 37.9 %, 2.50
tokens a pass), T 0.7 90.0 tok/s (41.7 %, 2.65); `drafts` diagnostic 57.7 / 57.9. Seeds (gate 3 on the test binary): **100 / 100
identical** at T 1.0, DFlash2 k4, off 0 passes vs on 326 passes (935 proposed, 140 accepted) (`raw/d1/seeds-*`).

`raw/d1/sim.txt` (`sim.py d1/cal --hip --k=4 --ref --base=<HIP recipe string>`):
- Cost table: D (keyed DFlash2 draft) 3.80 ms, P (plain step) 13.34 ms -> d 0.285; r_4 0.874 (464 passes; r_1..r_3 had < 10 passes,
  scaled): 15 % rule -> **measured table d=0.285, r=0.303/0.548/0.699/0.874** (built-in cut to 4: d 0.26, r 0.26/0.47/0.60/0.75).
- Replay check: 5 of 6 prompts x temperature exact, T 0.7 short_a not (51/204/75 replayed vs 46/181/66 measured: the DFlash2 draft's
  proposals are not the same between the fixed and the `drafts` run there); predicted ms within 0.93-1.04 of measured.
- The HIP recipe's policy parameters on that table miss the replay bars: short_b 0.964 (T 1.0) / below 0.99 (T 0.7), T 1.0 4k class
  1.339 < 1.437. #543's grid, by the rule: best **g=0.85, margin=0.02, drop=0.05, probe=3, hold=16, hold_max=64**: classes T 1.0 short
  1.275 / 1k 1.075 / 4k 1.459, T 0.7 short 1.294 / 1k 1.188 / 4k 1.435 (all >= 0.98 x DSpark-adaptive), lowest prompt **0.966**
  (short_b, T 1.0) - the replay predicts bar 1 (every prompt >= 0.98) fails at short_b; D2 measures it.
- **DFlash2 table for D2 / PR B:** `d=0.285,r=0.303/0.548/0.699/0.874,g=0.85,margin=0.02,drop=0.05,probe=3,hold=16,hold_max=64`.

## D2: sampled DFlash2 k4 vs plain sampling (window D2 #12485, 2:38, binary specB1) - **gate FAILS**

Every `pol` process: `--spec dflash --spec-draft /models/Inkling-2-Mini-DFlash2 --spec-k 4: the DFlash2 default of recipe inkling2 on
hip` and `draft length per request on (d=0.285,r=0.303/0.548/0.699/0.874,g=0.85,...,margin=0.02,drop=0.05,probe=3,...)` (the D1
table); `off` = `INTARSIA_SPEC_SAMPLED_DFLASH=0`, same binary. 4 reps per card (card 0 T 1.0, card 1 T 0.7), ids identical across all
8 processes per temperature. `raw/d2/analysis.txt` (`speed3.py d2 --ref`).

| T | class | plain tok/s | sampled DFlash2 tok/s | DFlash2 / plain [95 % CI] | bar: 0.98 x DSpark-adaptive (#543) | |
|---|---|---:|---:|---:|---:|---|
| 1.0 | short | 75.87 | 98.57 | 1.299 [1.265-1.334] | 1.149 | ok |
| 1.0 | 1k | 73.43 | 83.78 | 1.141 [1.122-1.160] | 1.021 | ok |
| 1.0 | 4k | 72.32 | 98.52 | 1.362 [1.350-1.374] | 1.437 | **FAIL** |
| 0.7 | short | 76.05 | 91.16 | 1.199 [1.170-1.228] | 1.140 | ok |
| 0.7 | 1k | 73.91 | 78.40 | 1.061 [1.053-1.069] | 1.047 | ok |
| 0.7 | 4k | 72.62 | 97.97 | 1.349 [1.337-1.362] | 1.429 | **FAIL** |

Per prompt (bar 1, >= 0.98 each): T 1.0 short_b **0.974** [0.949-1.000] (FAIL; acceptance 22.4 %), short_a 1.363, short_c 1.831,
len1k 1.141, len4k 1.362; T 0.7 short_b 0.990, short_a 1.054, short_c 1.849, len1k 1.061, len4k 1.349. Seeds (bar 3, D1): 100 / 100.

**Verdict: (b) fails bars 1 and 2** (short_b at T 1.0 as the D1 replay predicted, 0.966; and 4k at both temperatures, where
sampled DFlash2 gains 1.35-1.36x against DSpark-adaptive's 1.46-1.47x). As pre-registered: no re-tune, no re-run; reported to the
coordinator (00:10). Flip B is not shipped; the PR B candidate (resolver, keys, tests) is kept unpushed for (a).

## PR B candidate `specB1` (built before D2, 23:55-00:00)

PR A head `136b4dc16` (main `1bf40e29f` with #544 merged in) + `raw/prb_patch.py` + `raw/prb_patch2.py` with the D1 table, applied in
tatooine's lane folder while the Mac worktree stayed on PR A for its cpu suite; hip CLI md5 `c021c060c03f30cee4a2a54fc1df1cd7`.
- **Rule:** new recipe keys `[tuning.<backend>] dflash2_draft` (`"Inkling-2-Mini-DFlash2"` on hip), `dflash2_k` (4),
  `dflash2_sampled_adapt` (the D1 table). Where the recipe's speculation default applies (no `--spec`, one sequence, one rank),
  `load::dflash2_default_draft` resolves the drafter: an absolute path, or a name next to the checkpoint (`<checkpoint>/../<name>`;
  on tatooine `/models/Inkling-2-Mini-DFlash2` beside `/models/Inkling-2-Mini`). Found = a DFlash2 `config.json` that `open_dflash`
  takes: the default becomes `--spec dflash --spec-draft <it> --spec-k 4`, one drafter loaded, and the model's `sampled_dflash`
  default covers it (with `dflash2_sampled_adapt`). Missing / not DFlash2: one stderr line, the DSpark default (flip A), never an error.
  `INTARSIA_SPEC_DFLASH2_DEFAULT=0`: the DSpark default.
- **Why a recipe field with a name next to the checkpoint:** no flag (the brief); the recipe already carries the per-backend default
  (`mtp_default_k`, `sampled_dflash`), so the drafter's name and k sit beside them; the sibling rule finds the drafter wherever the model
  directory is, as long as the drafter sits beside it (on tatooine both are in `/data/models/intarsia`), and an absolute path covers
  other layouts (an HF cache, where snapshots are not siblings). An env path would be a flag in another form.
- **cpu checks on tatooine (dev image, cpu backend):** `inkling2_dflash2` 10 / 10 (2 new: the resolver finds a sibling / an absolute
  drafter, `=0` reverts, missing / target / DSpark dirs fall back, metal / cuda / cpu have no default; sampled requests draft with
  DFlash2 under the recipe key at k2 / k4 with `--spec off` ids at T 1.0 / 0.7, not vacuous, the DFlash2 table is used, no key keeps
  them plain), `recipe_ink_sampled_dflash` 3 / 3, `recipes_doc` 3 / 3, `inkling2_sampled_default` 2 / 2, cli `docs_env_vars` 2 / 2,
  core lib 284 / 284. Not yet the endor suite (gate 1).

## Option (a) as built (PR B, branch `ink/hip-specflip-b`)

- **Rule.** Where the recipe's DSpark default applies (no `--spec`, one sequence, one rank, no prefix store: Inkling-2 on HIP), the
  recipe's `[tuning.hip] dflash2_draft = "Inkling-2-Mini-DFlash2"` is resolved (`load::dflash2_default_draft`: an absolute path, or a name
  next to the checkpoint, `<checkpoint>/../<name>`; found = a DFlash2 `config.json` that `open_dflash` takes). Found and the checkpoint
  holds its DSpark draft: DFlash2 loads as a **second draft** (`LoadOptions::dflash_alt`, k `dflash2_k` = 4) beside the DSpark draft
  (k5). Greedy requests draft with DFlash2 k4, sampled requests with DSpark (flip A's `sampled_dflash` and adaptive table, unchanged).
  Found but no DSpark draft: DFlash2 alone is the default (greedy drafts, sampled plain). Not found / not DFlash2 / fails to open, or
  `INTARSIA_SPEC_DFLASH2_DEFAULT=0`: the DSpark default alone, PR A's load (DFlash2 not read). Never an error.
- **Engine.** `Model` holds `dflash` (DSpark) and `dflash_alt` (DFlash2); each `SequenceState` holds the active draft state (`dflash`)
  and the other one parked (`dflash_parked`, `dflash_alt_on` says which). `Seq::new` calls `Model::select_dflash(st, greedy)` for a
  drafting request: a swap parks the old state, starts the incoming one over (empty; synced only on an empty sequence) and drops the
  captured graphs (the target's taps go to the other draft's buffers). The existing resync rule then applies as after a turn that did
  not draft (`DFLASH_RESYNC_REUSE_MAX` = 128: a short reuse re-prefills, a longer one runs that request without drafting). Only the
  active draft ingests each pass, so holding both costs memory, not time. Every site that read `Model::dflash` for a sequence reads the
  state's draft (`Model::dflash_of(st.dflash_alt_on)`: taps in the Inkling-2 forward, ingest, draft, host ingest); the plan dry run
  records both drafts' passes; verify rows cover the larger k (k5 + 1).
- **Memory.** The DFlash2 draft shares the DSpark draft's mxfp4 copy of the unembed (`INTARSIA_DSPARK_HEAD` and `INTARSIA_DFLASH2_HEAD`
  both mxfp4 on HIP: the same rows of the same tensor through the same `quantize_host`, identical bits), about 0.25 GB not spent twice;
  otherwise its own copy as before.
- **Why a recipe field with a name next to the checkpoint** (the brief: no flag): the recipe already carries the per-backend default
  (`mtp_default_k`, `sampled_dflash`), so the drafter's name and k sit beside them; the sibling rule finds the drafter wherever the model
  directory is, as long as the drafter is downloaded beside it (tatooine: both in `/data/models/intarsia`), and an absolute path covers
  other layouts (an HF cache, where snapshots are not siblings). An env path would be a flag in another form.
- **Unchanged:** Metal, CUDA, cpu (no `dflash2_draft` there), `--spec off`, explicit `--spec`, `--max-batch` > 1 (the DSpark default is
  off above one sequence, so neither draft loads: the batching engine takes plain steps as on main).
- **Revert:** `INTARSIA_SPEC_DFLASH2_DEFAULT=0` (the DSpark default alone, flip A), or delete the two recipe lines.
- **cpu tests** (`tests/inkling2_dflash2.rs`): the resolver (sibling / absolute found, `=0` off, missing / target / DSpark dirs fall back,
  no default on metal / cuda / cpu); a fresh greedy request on the dual engine drafts exactly as DFlash2 k4 alone (same passes,
  proposals, acceptances, ids) and a fresh sampled one exactly as DSpark k5 alone; one conversation of greedy / sampled turns (T 1.0,
  0.6) emits `--spec off`'s ids; the dual load parks DFlash2's state and its context is <= the DSpark-only one, which is the same
  without the second draft; DFlash2 alone keeps sampled requests plain; a DSpark second draft is refused.

**(a) window plan (sent ~00:42 EST by the Mac clock; the message said 00:58), approved with changes 1-4 below:** control `specA2` (PR A). WA1 ~4:40: KV first (card 1; skip the speed
part if worse than -10 %), card 0 greedy ABBA half 1 (dfl dual dual dfl), card 1 md5 off / default + 100 seeds T 1.0; WA2 ~3:50: card 0
greedy ABBA half 2 (dual dfl dfl dual), card 1 100 seeds T 0.6 + 64-token md5; WA3 ~4:50: sampled ABBA both cards (T 1.0 / 0.7, n = 4,
dual vs specA2 default), then b16 serve of the dual binary vs PR A's GS2 b16 replies. ~13:20 in all (lane ~41:57 of 45). Proposed
deviation: the serve check for (a) is dual b16 = PR A b16 (b16 loads no drafter: unchanged from main), not dual solo vs b16 (a cold solo
admission alone needs a 4th window, ~46:17 > 45); solo correctness = md5 + seeds.

## Pre-registration: the (a) gate (written 00:25 EST 11-27, before any (a) window; coordinator 00:08 as written above)

(a) = both drafters loaded on HIP at one sequence: greedy requests draft with DFlash2 k4, sampled requests with DSpark (adaptive, flip A's
table). Fallbacks: DFlash2 missing or `INTARSIA_SPEC_DFLASH2_DEFAULT=0` -> exactly PR A (DFlash2 not loaded: same head count, same KV
line); DSpark draft missing -> greedy DFlash2, sampled plain. `--max-batch` > 1 loads no drafter (as main). Only the drafter of the
request ingests each pass; the other goes unsynced and the existing resync rule (`DFLASH_RESYNC_REUSE_MAX`) applies when it is picked again.

Gate (every item must pass; no re-tune, no re-run hoping for a pass):
1. **Ids.** Greedy ids of the dual default = `--spec off` (md5.json, 256 tokens; and the 64-token golden prompt). 100 / 100 seeds equal
   `--spec off` at **T 1.0 and T 0.6** (not vacuous: verify passes > 0 on the default side).
2. **Speed (contention from holding both).** Control = `specA2` (PR A head, one drafter): greedy = `--spec dflash --spec-draft <DFlash2>
   --spec-k 4` (DFlash2 alone), sampled = its default (DSpark adaptive). Dual binary at its default for both. Greedy cells = #541's six
   (short, 1k, prose, chat, 4k, 16k), ABBA 4 + 4 fresh processes; sampled cells = #543's prompts (short_a, short_b, short_c, len1k, len4k)
   at T 1.0 and T 0.7, n = 4. **Every cell's dual / control ratio has a 95 % CI upper bound >= 0.99.**
3. **KV.** Measured solo capacity (`--max-seq 1000000`) with both loaded; **stop and report if it is worse than -10 %** of DSpark-alone
   (177,504 on card 1); the default solo context 4096 must still load. Measured first, in the first (a) window; the stage skips its speed
   part if this fails.
4. **Tests.** Full cpu suite (endor) and the hip tiny tests pass on the PR head; engine tests: a missing drafter / the revert env gives a
   load identical to PR A (head count, KV line); a greedy turn then a sampled turn (and back) in one conversation keeps ids = `--spec off`.
5. Serve 16 / 16 at `--max-batch 16` = solo; Metal no-op check (Metal greedy md5 main vs branch, Metal tiny suite).

## Pre-registration: the sampled DFlash2 test (option (b); written ~23:20 EST, before windows D1 / D2)

**D1 (calibration, #543's W1 protocol), binary `specA1`** (CLI of d84acc9f6; sampled drafting with `INTARSIA_SPEC_SAMPLED_DFLASH=1` and
any `--spec-draft`): `generate --spec dflash --spec-draft /models/Inkling-2-Mini-DFlash2 --spec-k 4`, #543's `speed.json`
(`raw/speed-adapt.json`, md5 `0c1a62af18620f63255d4036c6843a40`: warm, short_a / short_b / short_c, len1k, len4k), 128 tokens,
`--ignore-eos --top-k 0 --top-p 1.0 --seed 11`, `INTARSIA_SPEC_TRACE=1`; modes `0` (fixed k4) and `drafts`; card 0 T 1.0, card 1 T 0.7,
one process at a time per card (`raw/stage_d1.sh`). Card 0 then (no timed run on it): 100 seeds at T 1.0, 24 tokens, DFlash2 k4,
drafting off vs on (gate 3, on the test binary; repeated on the PR B binary in its gate window).

**Calibration rule (`raw/sim.py`, #543's with `--k=4 --ref --base=<the HIP recipe string>`):** the cost table (d, r_1..r_4) from the
traces by #543's 15 % rule against the HIP built-in table cut to 4 rows (a measured table where any value is > 15 % off); policy
parameters start from the HIP recipe's (`g=0.95,margin=0.0,drop=0.05,probe=1,hold=16,hold_max=64`). If the offline replay of that
policy has every prompt pol / off >= 0.99 and every class >= 0.98 x the DSpark-adaptive ratio below, it is the DFlash2 table; else #543's
grid (g x margin x drop x probe x hold x hold_max) keeping every class at that bar and maximising the lowest prompt. No other tuning.

**D2 (measurement), the PR B candidate binary** (built before D2 so its admission cache is warm for the gate): arms `off` (plain
sampling: `INTARSIA_SPEC_SAMPLED_DFLASH=0`) and `pol` (the PR B default: DFlash2 found, sampled DFlash2 k4 with the D1 table), same
binary and flags, #543's W2 protocol: speed-adapt.json, 128 tokens, `--ignore-eos --top-k 0 --top-p 1.0 --seed 7`; card 0 T 1.0, card 1
T 0.7; 4 reps per card in A B B A / B A A B order (8 processes per card). Analysis `raw/speed3.py` (#543's): per prompt and per class
(short pooled) pol / off, geometric mean over reps, 95 % t-interval.

**Gate (the coordinator's, unchanged):** (1) every prompt's pol / off >= 0.98 at both temperatures, short_b included (point estimate);
(2) every class's pol / off >= 0.98 x #543's DSpark-adaptive ratio on HIP: T 1.0 short 1.172 / 1k 1.042 / 4k 1.466 (bars 1.149 / 1.021
/ 1.437), T 0.7 short 1.163 / 1k 1.068 / 4k 1.458 (bars 1.140 / 1.047 / 1.429); (3) 100 seeds at T 1.0 equal to `--spec off`; (4)
greedy unchanged from the confirmed DFlash2 k4 (md5 gate). Fail: report <= 6 lines to the coordinator before any (a) code.

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

**Coordinator, 23:34 EST 11-26 (as written):**

> 1. #544 (int8 HIP, opt-in) is about to merge. It touches `model/inkling2.rs` and HIP `device.rs`. Before you build the PR B candidate / D2 binary, wait until `gh pr view 544 --json state` shows MERGED. Then merge `origin/main` into your branch (a merge, not a rebase) and build from that.
>    - If #544 is still open when you are ready to build, build anyway and say so in your report.
>    - PR A has to merge main too before it ships.
> 2. The user has gone to bed and said "keep working". Run everything already approved without waiting on me:
>    - PR A;
>    - the (b) test;
>    - PR B, if (b) passes.
>    If (b) fails, report to me in ≤ 6 lines before writing any (a) code.
> 3. A second lane, overlap stalls, may queue radiance windows later tonight. You keep priority: queue your windows as soon as each is ready.

**Lane to coordinator, 00:10 EST:** (b) fails its pre-registered gate (D2: short_b 0.974 at T 1.0; 4k 1.362 / 1.349 vs bars 1.437 /
1.429); no (a) code; PR A #545 continues (cpu suite #12480 running, HIP recheck window on the final-main binary next, then Metal).

**Coordinator, 00:08 EST 2026-11-27 (received ~00:13; as written):**

> Build (a) now: two drafters. The user already approved it as the fallback if (b) failed, knowing the −9.9 % solo KV cost, and did not say "stop at (b)". No need to wait for the morning.
>
> Order:
> 1. PR A first. The HIP recheck window (~4:30) on the final-main binary is approved, then the Metal slot. Report when #12480 and the recheck are done; I merge #545 on green CI.
> 2. Write the (a) code while those jobs run. Start from the unpushed PR B candidate, rebased on PR A. Greedy requests use DFlash2 k4. Sampled requests use DSpark with adaptive draft length. If either drafter file is missing, fall back to what's present (DSpark-only = flip A behaviour, DFlash2-only = greedy DFlash2 and sampled with no drafting). `--max-batch` > 1 still loads no drafter.
> 3. Pre-register the (a) gate in the results doc before measuring:
>    - Greedy ids equal `--spec off` (golden 256/64-token md5) and 100/100 seeds equal off at T 1.0 and 0.6.
>    - Speed: each path with both drafters loaded vs its single-drafter binary (greedy vs DFlash2-only k4, sampled vs PR A), every cell ≥ 0.99 by 95 % CI upper bound. This catches contention from holding both.
>    - KV: record measured solo capacity with both loaded; stop and report if it's worse than −10 %. Default solo context 4096 must still load.
>    - cpu suite and hip tiny tests pass on the PR head.
> 4. Radiance: 21:51 used + 4:30 recheck leaves ~6.5 min of the 33 cap. If the (a) gate needs more, send me the window plan with minutes first; I'll grant up to a 45-min lane total. Windows ≤ 5 min each, none starting after 06:00, never cancel a running window.
> 5. Write this go-ahead (00:08 EST, build (a), gate above, the 45-min ceiling) into the results doc now, so it survives compaction.
>
> When you report (a), give the speed table (greedy and sampled cells), md5/seed results, KV numbers, PR number and head.

**Lane to coordinator, 00:26 EST:** cpu suite #12480 on `136b4dc16` green (core-lib 283, scheduler 53, models 629, cli 208); GA2 md5 /
seeds / contexts pass, serve 16 / 16 not run (solo cold admission > 175 s); asked for one serve-only rerun (~4:30).

**Coordinator, 00:27 EST 11-27 (as written):**

> Approved: one serve-only rerun window for PR A on 136b4dc16, 285 s cap, solo first and alone with up to 240 s for cold admission, then b16 (warm).
>
> - One try only. If solo admission runs past 240 s again or the window voids for any other reason, don't queue a third. Report, and I decide whether G2c's 16/16 on specA1g plus the opt-in nature of #544 is enough to merge.
> - If it passes (b16 replies equal solo 16/16), tell me and I merge #545.
> - Radiance after it: ~30:30 of 33. The 45-min ceiling for (a) still stands, on a window plan first.
> - Write this approval (00:27 EST, one try, 285 s) into the results doc.
>
> Metal binaries and the Metal slot go ahead as planned.

**Lane to coordinator, ~00:42 EST (Mac clock; the message said 00:58):** the (a) window plan above (WA1 / WA2 / WA3, ~13:20) and the
serve deviation.

**Coordinator, 00:43 EST 11-27 (Mac clock; corrected from 00:59 by the coordinator), as written:**

> Go on the (a) window plan once the dualA1 build and cpu tiny tests pass, with these changes:
>
> 1. Greedy control is specA2's default (DSpark k5), not DFlash2-only. The user approved flip B on the GO rule vs DSpark k5, so apply that rule to dual directly: mean of #541's six cells ≥ 1.03, and no cell's 95 % CI upper bound below 0.99. At 4k, dfl6 was 0.991 [0.987-0.995], so a 1 % dual-vs-dfl loss could break GO there; measuring vs dfl-only wouldn't catch that. Same ABBA shape (A=specA2, B=dual, split over WA1/WA2, n=4). Sampled stays as planned: dual vs specA2, every cell ≥ 0.99 by CI upper bound.
> 2. Exact acceptance check (untimed, from existing runs where possible): on the greedy md5 prompts, dual's verify-pass and accepted-token counts must equal DFlash2-only k4 (`--spec dflash` k4, #541's dfl6 runs). On the 100 T 1.0 seeds, dual's counts must equal specA2's. That proves the shared mxfp4 unembed gives identical drafts and that parking doesn't change them. Any mismatch stops the flip.
> 3. Mixed-switch test, because WA1-WA3 never switch request type in one process:
>    - A cpu tiny test that alternates greedy and sampled requests on one engine, with ids equal `--spec off` and no panic on swap.
>    - An untimed HIP run on card 1 in WA2: one process, about 8 requests alternating greedy and seeded-sampled over a shared prefix, ids equal off.
> 4. Serve deviation approved: dual b16 = PR A GS2 b16, 16/16. b16 loads no drafter, and dual-solo correctness comes from md5 + seeds vs off. Record it in the results doc as a deviation with that reason.
>
> Ceiling stays 45:00. If a window voids or runs long and a rerun would cross 45, report before queuing. No window starts after 06:00.
>
> Write this go-ahead (00:59 EST, changes 1-4) into the results doc before queuing WA1.
>
> When you report, give: the GO table (dual / DSpark k5 with CI, six cells + mean), the sampled table, md5/seeds, acceptance-count equality, the mixed-switch result, KV numbers, PR number and head.

**Coordinator correction (as written):** "record the (a) go-ahead as 00:43 EST, not 00:59. The Mac clock read 00:43 when I sent it, and
your last message was stamped 00:58. Take timestamps for the results doc from `date` on the Mac." (Lane: from here on every time in this
doc is the Mac's `date`.)

**Deviation (approved, change 4): the (a) serve check** is the dual binary's `--max-batch 16` replies = PR A's GS2 b16 replies, 16 / 16,
not dual solo vs b16: above one sequence the DSpark default is off, so no drafter loads (the batching engine is the one PR A and main
run), and the dual solo engine's correctness is shown by its greedy md5 and seeds against `--spec off`.

**How changes 1-3 are run (amends the pre-registration's item 2 and adds items 6-7, written before WA1):**
- Greedy GO arms: `dspark` = `specA2` default (DSpark k5), `dual` = the (a) binary's default; WA1 card 0 A B B A (r1 r2), WA2 card 0
  B A A B (r3 r4); `speed.py <dir> dspark "short=1-12,1k=13,4k=14,16k=15,prose=16-17,chat=5-12,code=3+18"`; GO = geometric mean of the
  six cells >= 1.03 and no cell's CI upper bound < 0.99. Sampled: unchanged (dual / specA2 default, every cell's CI upper bound >= 0.99).
- 6. **Acceptance counts equal** (any mismatch stops the flip): md5.json at 256 tokens, the dual default vs `specA2 --spec dflash
  --spec-draft <DFlash2> --spec-k 4` (run in WA1 on card 1: no DFlash2-only md5 run exists from #541): `spec: verify passes / proposed /
  accepted` and every per-prompt `spec[...]` line equal; 100 seeds at T 1.0: the dual default's `spec verify passes / proposed /
  accepted` = GA2's specA2 default (265 / 408 / 123).
- 7. **Mixed switch on HIP** (WA2, card 1, untimed): the seed harness with `INTARSIA_SS_TEMPS=0.0,1.0`, seeds 0..4 (8 requests in one
  process, greedy and seeded sampled alternating over the same prompt, so every request swaps drafts), once with the short prompt
  (reuse <= 128: the incoming draft re-prefills) and once with it repeated 12 times (reuse > 128: that request runs without drafting);
  dual default vs no speculation at all (`INTARSIA_SPEC_DEFAULT=0`): ids equal, and the dual runs drafted (verify passes > 0).

**WA1 (#12517, 00:54:18-00:57:00, 2:42; dualA2) partly void** (`raw/a1-dualA2/`, `raw/window-wa1-12517.log`): kept: greedy md5.json 256
tokens off = dual default = DFlash2 k4 alone (`380af8a5`), acceptance dual = DFlash2 k4 alone (140 / 555 / 370, every per-prompt spec line
equal). Void 1 (KV): the dual load at `--max-seq 1000000` ran out of memory in `LocalEngine::new`'s first `new_state` (hipMalloc of 335 MB:
dualA2 allocated the DSpark state and a parked DFlash2 state, both at full capacity); fix dualA3: one draft state at a time
(`select_dflash` frees the old state and allocates the other draft's; the KV budget still counts both drafts' per-token cost, so the swap
always fits). Void 2 (seeds): 0 verify passes on both arms: the harness's default mode left `EngineOptions::spec_auto` false (the CLI sets
it when no `--spec` is given), so no drafter loaded; fixed (dualA4 = dualA3's CLI, md5 `579b3042`, + the fixed harness), and the stage marks
a default run without the DFlash2 line VOID. Lane to coordinator 01:01 (rerun plan ~44:09, stop rule: one more overrun or void -> stop and
report).

**Coordinator, 01:03 EST 11-27 (as written):**

> Go on WA1b → WA2 → WA3 as planned (~44:09). Both fixes are sound, and the sampled acceptance compare against `INTARSIA_SPEC_DFLASH2_DEFAULT=0` on the same binary and harness path is fine as the like-for-like version of change 2.
>
> Since dualA3 now frees and reallocates the draft state and drops graphs on every greedy↔sampled switch, add two things to the untimed mixed-switch run on card 1 in WA2. No new window.
> 1. Per-request wall time, plus the time to first token of the first request after each switch, so the morning report can state the switch cost (hipMalloc + graph re-capture + resync).
> 2. Free VRAM (or the engine's own allocator stats) logged before the first request and after the last, with ≥ 8 switches in the run. It must not grow or creep toward OOM. A leak or creep stops the flip.
>
> Your stop rule stands: one more overrun or void and you stop and report rather than rerun. Write this note (01:03 EST) into the results doc.

How it runs (written before WA1b): the mixed run is seeds 0..6 x T (0.0, 1.0), seeds outer: 12 requests, each one a switch (12 swaps, the
first from the load's DSpark state); the harness prints per request its wall time and time to first token, and the backend's free memory
(`Backend::info().free_mem`) before the first request and after the last. Pass: ids = no speculation; free memory after >= before minus
64 MiB (allocator rounding); the switch cost is reported, not gated.

**WA1b (#12522, 01:08:34-01:12:48, 4:14; dualA5 = CLI md5 `579b3042` (= dualA3), harness `f04734975`; `raw/a1-dualA5/`):** KV:
capacity DSpark alone 177,504, both drafters **163,712 (-7.77 %)**, default context 4096 (gate 3: pass). Greedy md5.json 256: off = dual
= DFlash2 k4 alone (`380af8a5`); acceptance dual = DFlash2 k4 alone (140 / 555 / 370, every per-prompt line equal). Seeds T 1.0: 100 / 100
equal off; acceptance dual = DSpark alone (`INTARSIA_SPEC_DFLASH2_DEFAULT=0`) = 265 / 408 / 123 (also GA2's specA2 number), ids identical.
GO ABBA half 1 (card 0, A B B A): dspark 109.26 / 108.87 tok/s overall, dual 125.43 / 125.28; dual's verify counts per process (1265 /
5038 / 2935) = conf-k4's DFlash2 k4 alone.

**Coordinator, 01:14 EST 11-27 (as written):**

> #545 merged as 896198dcc (01:14 EST). Metal no-op #12515 checked: md5 main = PR A for off and default (d93147da, 169/429/311 acceptance both), sampled default line off on Metal, tiny suites pass.
>
> For (a):
> 1. Don't delete or reuse remote `ink/hip-specflip`. Further results-doc commits go on `ink/hip-specflip-b`, which must merge origin/main (896198dcc) before its PR.
> 2. #546 (Metal DFlash2, opt-in) merges next. It lifts Metal's refusal of DFlash2 in model.rs, so after it lands only the recipe key keeps Metal off the DFlash2 default. Add a test that `dflash2_default_draft(recipe, "metal", …)` and "cuda" return NotSet. #546 also edits `crates/intarsia-models/tests/inkling2_dflash2.rs`, which (a) changes too, so expect a conflict there when you merge main after it lands.
> 3. The (a) Metal no-op slot runs with the DFlash2 drafter present next to the checkpoint on endor (#12464 copied it there). The default must still load only DSpark k3 and greedy md5 must equal main's.
>
> No reply needed. Carry on with WA1b.

**PR A Metal no-op slot (#12515, endor exclusive, 01:00:17-01:09:58, 9:41; `raw/metal-specA2/`):** greedy md5 (2 prompts) main
`1bf40e29f` = PR A `8c4400fec` for `--spec off` and the default (`d93147da`, DSpark k3 169 / 429 / 311 both); the sampled default line
says off on Metal; Metal tiny suites inkling2_sampled_default 2 / 2, inkling2_dspark_sampled 2 / 2 (3 ignored), inkling2_dspark_sampled_adapt
2 / 2, inkling2_dspark 4 / 4. (#12509 before it exited at once: its process guard matched a shell line; the guard now matches only commands
that start with an intarsia binary.)

