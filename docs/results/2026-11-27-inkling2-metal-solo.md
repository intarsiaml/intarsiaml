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

(pending)
