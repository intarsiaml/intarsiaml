# Inkling-2 on Metal: greedy default DFlash2 k5 beside DSpark (flip M)

Lane ink-metal-flip, 2026-11-27. User decision 2026-11-27 ~12:10 EST: "Yes, flip Metal". The question put to the user was the Metal
greedy default DSpark k3 → DFlash2 k5, with the #549 numbers (decode 1.072x mean, greedy ids identical, +4.7 GB peak at the largest
`--max-seq`).

## Change

`recipes/inkling2.toml` `[tuning.metal]`: `dflash2_draft = "Inkling-2-Mini-DFlash2"`, `dflash2_k = 5`. No engine code. This is the
mechanism of flip B (#550), set for Metal: where the recipe's DSpark default applies (no `--spec`, one sequence, one rank), the DFlash2
drafter loads beside the checkpoint's own DSpark draft. Greedy requests draft with DFlash2 at k5. Sampled requests take plain steps, as
before, because sampled drafting is opt-in on Metal. Verify passes are sized for the larger k
(`crates/intarsia-models/src/load.rs:1346`), so k5 is not clamped by DSpark's k3.

What differs from what #549 measured: #549's arm `dfl5` was DFlash2 alone (`--spec dflash --spec-draft <DFlash2> --spec-k 5`). The
default now loads both drafts (DSpark idle for default requests on Metal). The gates below check that the two-draft load gives #549's
ids, speed and memory.

Drafter location on endor: the checkpoint is a Hugging Face snapshot, so `<checkpoint>/../Inkling-2-Mini-DFlash2` is
`.../models--thinkingmachines--Inkling-2-Mini/snapshots/Inkling-2-Mini-DFlash2`. A persistent link there points to
`../../models--incoai--Inkling-2-Mini-DFlash2/snapshots/4f9088ea66c21f2b99ed657d1941c306b03e84d2` (made 2026-11-27 12:16; it covers
every snapshot of the checkpoint). Without the link the default stays DSpark k3, with one stderr line.

Revert: `INTARSIA_SPEC_DFLASH2_DEFAULT=0` per run, or delete the two recipe lines, or remove the link.

## Gates (pre-registered 2026-11-27 ~12:17 EST, before the first slot; committed in 89b6e18e2)

One binary (this branch). Arms: `off` = `--spec off`; `main` = `INTARSIA_SPEC_DFLASH2_DEFAULT=0` (today's default, DSpark k3 alone);
`flip` = no `--spec` (the new default); `dfl5` = #549's arm. One endor `--exclusive` fleetq slot, script `raw/slotG.sh`.

| # | Check | Rule |
|---|---|---|
| G1 | Load | `flip` stderr has `DFlash2 drafter ... (k 5): loaded beside the DSpark default` and `dflash: second draft (greedy requests, k 5)`; `main` has `off (INTARSIA_SPEC_DFLASH2_DEFAULT=0)` |
| G2 | Greedy ids | 256 ids on `code`, `len4k`, `len16k` (`generate --ignore-eos`): md5 equal across `off` / `main` / `flip` / `dfl5`, and equal to #549's (`c69d1438...`, `9e85ba11...`, `89994b87...`) |
| G3 | Sampled ids | `flip` = `off` on every (config, prompt) pair: T 1.0 x 100, T 0.6 x 60, T 0.7 top-k 20 top-p 0.9 x 30, seed 7, 32 tokens; 0 verify lines in `flip`'s traced runs |
| G4 | Batching | `serve --max-batch 4`, no `--spec`: no draft loaded (no speculation line); 4 concurrent greedy replies = solo, twice |
| G5 | Memory | peak footprint at the clamped capacity (`--max-seq 4194304 --kv-fraction 0.25`), 2 runs per arm. **FLAG** if `flip` > `dfl5` + 0.5 GB (the two-draft load costs materially more than what the user approved) |
| G6 | Speed | ABBA `flip` / `main` on abba3.json (warm + 3 short prompts), 256 tokens, 45 s idle. **FLAG** if geo < 1.10 or any prompt < 0.95 (#549's `dfl5` / default: geo 1.253, min 1.028) |

Merge rule: G1-G4 PASS and no FLAG → PR, merge on green CI. A FAIL or FLAG → report to the user before merging.

**G7 (added 2026-11-27 12:38 EST after slot G, before slot H):** the one path not tested with both drafts loaded is `serve` at one
sequence (the default; `--max-batch` is opt-in) with multi-turn prefix reuse. #549's check 5 tested it with DFlash2 alone. Rule, as in #549's
check 5 (script `raw/slotH.sh`, fleetq slot H): `serve --max-batch 1` with the flip default vs `--spec off`, each started twice; 2 chats x 3
turns on the chat endpoint plus 2 x 3 turns as token-id extensions, greedy, 128 tokens. **PASS** if all 12 turns' ids are identical between
the flip server and the off server at both starts, and `cached_tokens` shows reuse on turns 2 and 3. A FAIL blocks the merge.

## Results

**Verdict: PASS. G1-G4 pass, and neither G5 nor G6 raised a FLAG.** Slot G was fleetq #12677, endor `--exclusive`, 12:18:31-12:32:24 EST, the
first and only slot. Binary `intarsia-flip` md5 `08b3211b60f82bceb080c795a030eb67`, built from 89b6e18e2 (same code as 6e9640a2c). The log
is `raw/slotG-12677.log`, and the per-run stderr is in `raw/G/`.

| # | Result | Key number |
|---|---|---|
| G1 | **PASS** | `flip`: `DFlash2 drafter .../snapshots/Inkling-2-Mini-DFlash2 (k 5): loaded beside the DSpark default of recipe inkling2 on metal` and `dflash: second draft (greedy requests, k 5) of 5 layers (... block 8, DFlash2)`. `main`: `DFlash2 drafter of recipe inkling2 on metal off (INTARSIA_SPEC_DFLASH2_DEFAULT=0): the DSpark default alone` |
| G2 | **PASS** | md5 equal across all 4 arms and equal to #549's: code `c69d1438ab3e715feb93a0614ebddcfe`, len4k `9e85ba1117e4a6a6cc12c5c917532e7f`, len16k `89994b876223335763b2004d0d281157`, 256 / 256 ids each. `flip`'s spec counters equal `dfl5`'s exactly (203 verify passes, 1008 proposed, 560 accepted, 3.76 tokens per pass); `main`: 272 passes, 2.63 tokens per pass |
| G3 | **PASS** | 190 / 190 (config, prompt) pairs `flip` = `off` (T 1.0: 100, T 0.6: 60, T 0.7 top-k 20 top-p 0.9: 30). `flip`'s traced runs: `spec: verify passes=0`, with startup line `sampled requests (temperature > 0): DFlash-family drafting off (default ...)`. Its other two "verify" lines are load-time configuration lines |
| G4 | **PASS** | `serve --max-batch 4`: up after 136 s (cold admission of a new binary), no spec / draft line in the log; concurrent = solo 4 / 4, twice |
| G5 | **no FLAG** | peak footprint at the clamped capacity: `main` 44.10 / 44.10 GB, `dfl5` 48.82 / 48.82, `flip` 48.98 / 48.97, so `flip` − `dfl5` = **+0.15 GB** (rule: +0.5). `flip` − `main` = +4.87 GB. Context 607,199 tokens vs 615,670 (`main`, −1.4 %) and 615,458 (`dfl5`). Weights 17.24 vs 16.41 GB (`main`). At `--max-seq 20480` (G2 runs): `off` 20.30, `dfl5` 20.32, `main` 20.84, `flip` 21.15 GB |
| G6 | **no FLAG** | ABBA `flip` / `main`: count **1.438** (1.390, 1.488), story **1.005** (0.979, 1.032), chat **1.284** (1.250, 1.318); geo **1.229**, min 1.005 (rule: 1.10 / 0.95; #549: 1.253 / 1.028). Ids identical across the 4 processes |

Single greedy runs, information only (G2's `generate` runs, one process each, no idle gaps). Decode in tok/s:

| prompt | off | main (DSpark k3) | flip | dfl5 |
|---|---|---|---|---|
| code | 77.6 | 106.5 | 136.4 | 141.5 |
| len4k | 68.6 | 90.6 | 104.7 | 91.3 |
| len16k | 63.7 | 67.1 | 66.2 | 66.4 |

Two notes:

- **The startup line's wording is HIP's.** The line `... for greedy requests (... sampled requests keep the DSpark draft; ...)` comes from
  `load.rs`. On Metal, sampled requests take plain steps, as the next line (`DFlash-family drafting off`) and G3 show. The message is
  cosmetic and was not changed here.
- **The flip adds weight without adding speed.** The DSpark draft (0.83 GB of weights in `flip` vs `dfl5`, −1.4 % context at the
  clamped capacity) is loaded but idle for default requests on Metal. It serves only `INTARSIA_SPEC_SAMPLED_DFLASH=1`. Loading DFlash2
  alone where sampled drafting is off would save that. That would be a code change, not done here.
