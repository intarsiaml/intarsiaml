# Inkling-2 on Metal: the DFlash2 default loads alone (the idle DSpark draft is not loaded)

Lane ink-metal-solo, 2026-11-27. A follow-up of flip M (#551, `docs/results/2026-11-27-inkling2-metal-dflash2-flip.md`).

## Why

Flip M loads the DFlash2 drafter beside the checkpoint's DSpark draft on Metal. Greedy requests draft with DFlash2. Sampled requests take
plain steps, because sampled drafting is opt-in on Metal (it failed its gate). So the DSpark draft is loaded but idle for every default
request. Flip M's G5 measured its cost: 0.83 GB of weights and -1.4 % context at the clamped capacity (`flip` vs `dfl5`).

## Change

`crates/intarsia-models/src/load.rs`: the recipe's DFlash2 default loads beside the DSpark draft only where sampled requests draft with
it (`dflash2_beside_dspark`: the checkpoint holds the DSpark draft AND `sampled_dflash_resolve(INTARSIA_SPEC_SAMPLED_DFLASH, recipe
sampled_dflash, own draft)` is on). Elsewhere it loads alone, through the existing "no DSpark draft in this checkpoint" path (`--spec
dflash --spec-draft <DFlash2> --spec-k <dflash2_k>`, the same as #549's `dfl5` arm).

| Backend | Before | After |
|---|---|---|
| Metal, default | DFlash2 + idle DSpark | **DFlash2 alone** |
| Metal, `INTARSIA_SPEC_SAMPLED_DFLASH=1` | DFlash2 + DSpark | DFlash2 + DSpark (unchanged) |
| HIP, default (sampled drafting on, flip A) | DFlash2 + DSpark | DFlash2 + DSpark (unchanged) |
| HIP, `INTARSIA_SPEC_SAMPLED_DFLASH=0` | DFlash2 + idle DSpark | DFlash2 alone |

Tokens cannot change: greedy requests draft with the same DFlash2 draft at the same k, and sampled requests take plain steps in both.
Revert: `INTARSIA_SPEC_SAMPLED_DFLASH=1` loads both drafts again (and turns sampled drafting on); or revert the PR.

## Gates (pre-registered 2026-11-27 ~20:25 EST, before the first endor slot)

One binary (this branch), one endor `--exclusive` fleetq slot, script `raw/slotS.sh`. Arms: `solo` = no `--spec` (new default);
`both` = `INTARSIA_SPEC_SAMPLED_DFLASH=1`; `off` = `--spec off`; `dfl5` = `--spec dflash --spec-draft <DFlash2> --spec-k 5`.

| # | Check | Rule |
|---|---|---|
| S1 | Load | `solo` stderr: `sampled drafting is off on this backend, so the DSpark draft is not loaded`, one `dflash: draft of 5 layers ... DFlash2` line, no `second draft` line. `both`: `loaded beside the DSpark default` and `second draft (greedy requests, k 5)` |
| S2 | Greedy ids | 256 ids on `code`, `len4k`, `len16k`: md5 equal across `solo` / `both` / `off` / `dfl5` and equal to #549's (`c69d1438...`, `9e85ba11...`, `89994b87...`); `solo`'s `spec: verify` counters equal `dfl5`'s |
| S3 | Sampled ids | `solo` = `off` on every (config, prompt) pair: T 1.0 x 100, T 0.6 x 60, seed 7, 32 tokens; `solo` traced: 0 verify passes |
| S4 | Memory | `solo`'s weights (`loaded` line) at least 0.7 GB below `both`'s; at the clamped capacity `solo`'s context above `both`'s, both runs |
| S5 | Serve | `serve --max-batch 1`, `solo` default vs `--spec off`: all 12 turns (2 chats x 3 on the chat endpoint, 2 x 3 token-id extensions, greedy, 128 tokens) identical; the `solo` server log shows DFlash2 alone |

Merge rule: S1-S5 PASS → PR, merge on green CI. Any FAIL → report, merge nothing.

Not re-measured: decode speed. `solo` runs the same draft, k and verify plan as `dfl5`, which #549 and flip M measured (S2's counters
check that it does the same work).

## Results

Slot S: fleetq #12804, endor `--exclusive`, 20:25:54 to 20:35:41 (587 s). Binary `intarsia-solo` md5 `46b2660b…` (`bb4873cec`).
Log: `raw/slotS-12804.log`; per-run files: `raw/S/`.

| # | Result | Evidence |
|---|---|---|
| S1 | **PASS** | `solo`: "sampled drafting is off on this backend, so the DSpark draft is not loaded", one `dflash: draft of 5 layers ... DFlash2` line, no second draft. `both`: "loaded beside the DSpark default", a 3-layer DSpark draft plus "second draft (greedy requests, k 5)" |
| S2 | **PASS** | 256 ids, md5 equal across `solo` / `both` / `off` / `dfl5` and equal to #549's: code `c69d1438…`, len4k `9e85ba11…`, len16k `89994b87…`. `solo` = `dfl5` = `both` counters: 203 passes, 1008 proposed, 560 accepted |
| S3 | **PASS** | `solo` = `off` on 160 of 160 (T 1.0 x 100, T 0.6 x 60, seed 7, 32 tokens); `solo` 0 verify passes |
| S4 | **PASS** | Weights 16.27 GB `solo` vs 17.24 GB `both`: **0.97 GB less**. Context at the clamped capacity 615458 vs 607199 tokens (**+1.36 %**), both runs of each arm. Peak footprint (S1 runs, `--max-seq 20480`) 20.27 vs 21.16 GB |
| S5 | **PASS** | `serve --max-batch 1`: 12 of 12 turns identical to `--spec off` (2 chats x 3 on the chat endpoint, 2 x 3 token-id extensions), `cached_tokens` equal per turn. The `solo` log shows DFlash2 alone |

- **Slow first server start (195 s):** the `solo` server was this binary's first `serve`. Its plan-shape admission ran uncached
  ("42 kernel(s) checked, 169 case(s) run, 0 cached"). Flip M had the same pattern: first start 141 s, second 8 s
  (`docs/results/2026-11-27-inkling2-metal-dflash2-flip/raw/slotH-H.log`). This is not caused by this change.
- **Decode speed** was not a gate. The S1 runs ran back to back with no idle gaps (endor's GPU burst), so their tok/s are not a
  measurement.
