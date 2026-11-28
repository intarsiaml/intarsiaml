# Inkling-2-Mini on Metal: DFlash2 k7 + confidence cut 0.3 and the prefill prologue split by default (lane ink-metal-flip2, 2026-11-27)

Status: **gates pre-registered, slot F not run yet.**

User decision 2026-11-27 ~23:05 EST, "Flip both", on the two opt-in levers measured today:

| Lever | Opt-in result | PR |
|---|---|---|
| DFlash2 `--spec-k 7` + `INTARSIA_DFLASH_CUT=0.3` (greedy decode) | 1.126x DFlash2 k5 over the six cells, ids identical, slot R ready gates pass (`2026-11-27-inkling2-dflash-cut.md`) | #556 |
| Prefill KV prologue split `INTARSIA_INK_PROLOGUE_SPLIT=1` | bitwise, prefill +3.1-3.3 % at 1k / 4k / 16k (`2026-11-27-inkling2-metal-prologue-split.md`) | #555 |

## The change (branch `ink/metal-flip2`, on main `e76cbd89b`)

Two commits, so either can be reverted alone:

1. `4f306174c`: the prologue split is on unless `INTARSIA_INK_PROLOGUE_SPLIT=0` (`crates/intarsia-backend-metal/src/inkprefill.rs`). Metal only
   (the kernel exists only there), prefill chunks of more than 16 rows.
2. `30420f61c`: the DFlash2 cut defaults to 0.3 on Metal when `INTARSIA_DFLASH_CUT` is unset or blank (`dflash2.rs`
   `CUT_DEFAULT_METAL`, `cut_resolve`; `=0` turns it off; HIP, CUDA and cpu stay off unless set), and `recipes/inkling2.toml`
   `[tuning.metal] dflash2_k = 7` (was 5). The tiny-fixture DFlash2 tests set `INTARSIA_DFLASH_CUT=0`: the random model's 16
   near-uniform candidates (~1/16 each) would cut every draft and make the drafting checks vacuous on `INTARSIA_INKLING2_BACKEND=metal`.

Revert: `INTARSIA_INK_PROLOGUE_SPLIT=0`; `--spec-k 5` (or `dflash2_k = 5`) with `INTARSIA_DFLASH_CUT=0`.

## Gates (pre-registered before slot F)

One binary (the branch), one endor `--exclusive` fleetq slot, script `raw/slotF.sh`. Arms: `flip` = no flags, no env (the new
default); `main` = today's main default through the revert switches (`--spec-k 5`, `INTARSIA_DFLASH_CUT=0`,
`INTARSIA_INK_PROLOGUE_SPLIT=0`); `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| F0 | Prologue split bitwise on the GPU | `ink_prologue_split` test exe passes (built from the branch) |
| F1 | Greedy ids + what loaded | g3.json (code, len4k, len16k), 256 tokens, `--ignore-eos`: md5 `off` = `main` = `flip` = the goldens (`c69d1438...`, `9e85ba11...`, `89994b87...`). `flip` loads DFlash2 at k 7 with "dflash: cut 0.3 (the Metal default ...)" and traces `prologue split`; `main` shows k 5, no cut line, no split trace |
| F2 | Sampled ids | seed 7, 32 tokens, T 1.0 (100 prompts) and T 0.6 (60): `flip` = `off` on every prompt |
| F3a | Decode | ABBA x 2 over speed.json (`raw/rule.py flip main`): mean of the six cells >= 1.03, ids identical in every process |
| F3b | Prefill | ABBA x 2 over pf3.json, 1 token (`raw/pfabba.py`, off = main, on = flip): on / off of the medians >= 0.98 at each prompt |
| F4a | Serve, one sequence | `serve --max-batch 1`, flip (no `--spec`) vs `--spec off`: 12 / 12 turns' ids and `cached_tokens` equal |
| F4b | Serve, batching | `serve --max-batch 4` (flip): solo = concurrent x 2, and solo = #551's committed batch-4 replies (`2026-11-27-inkling2-metal-dflash2-flip/raw/G/b4-solo.json`, before the split existed) |

F3 is a confirmation with two reps per arm, not a new A/B: both levers were measured on their own with more reps (#555: 6 runs per
arm; #556: 4 reps). Its bars only catch a regression from combining them. 45 s GPU idle before every timed run, macmon logged.

**Decision rule:** all pass, so a PR and merge on green CI (the user decided the flip). Any F1, F2 or F4 ids mismatch: no merge, report.
F3 below its bar: no merge, report the numbers and ask.
