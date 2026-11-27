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

## Gates (pre-registered 2026-11-27 12:25 EST, before the first slot)

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

## Results

(pending)
