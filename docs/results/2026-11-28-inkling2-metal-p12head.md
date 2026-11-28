# Inkling-2-Mini on Metal: the DFlash2 draft head on the target's P12 unembed (lane ink-metal-p12head, 2026-11-28)

Status: **slot H (fleetq #12947, 2026-11-28 00:55-01:10) passes H1-H4: GO by the pre-registered rule.** Decode +1.6 % in matched
power (geo 1.020 [1.008-1.033] by the rule), bitwise (same candidates, counts and ids). Opt-in (`INTARSIA_DFLASH2_HEAD=p12`); making it
the Metal default is the user's call.

## Why

The DFlash2 draft scores its candidates with the target's unembed. On Metal it reads the **bf16** copy:
`logits_bf16_gemv_rows 200058x2304` at 7 rows takes 1.71 ms a pass at 539 GB/s (slot P of lane ink-metal-dtopk). That is already at
bandwidth, so only fewer bytes can make it faster.

The target's own verify head reads its **lossless P12 copy** of the same tensor (`matmul_p12_logits_gemv_rows`, lane ink-pack12,
default on on Metal):
- 75 % of the bytes (692 vs 922 MB).
- Bitwise the bf16 head at rows 1..16 at the real shape [200058, 2304], rows-exact included
  (`crates/intarsia-backend-metal/tests/p12.rs`).
- Faster than bf16 at every measured row count (`2026-11-21-inkling2-pack12.md`).

Lane ink-metal-dgemv showed that the per-dispatch profile cannot rank levers. This one cuts bytes on a bandwidth-bound kernel.
Expected: about 0.4-0.5 ms of a ~28.7 ms pass, about 1.5 % decode, with the same candidates.

## Change

`crates/intarsia-core/src/model/dflash.rs`, the DFlash2 draft head:
- With `INTARSIA_DFLASH2_HEAD=p12`, the draft head of the unsplit draft takes `Model::head_p12` when all of these hold:
  - the draft has no head copy of its own (`full_head` is `None`);
  - the copy is P12 (not HIP's i8);
  - t - 1 <= `p12::ROWS_MAX`;
  - the registry selects the call.
- Otherwise it takes the bf16 head as before.
- `exact_rows` follows the pass (`rows_k` sets it), so the call takes the rows-exact P12 kernel like the target's verify head.
- One stderr note per process names the kernel taken, or the fallback.
- `dflash2_head_kind` accepts `p12` (no copy).
- The shard, full-head and DSpine paths are unchanged.

## Checks (pre-registered before slot H)

One binary (`intarsia-ph`: main `919f52ee7` + this change), endor `--exclusive`, `raw/slotH.sh`. Arms:
- `dflt` = the Metal default (DFlash2 k7 + cut 0.3, the draft head on bf16).
- `p12` = `INTARSIA_DFLASH2_HEAD=p12`.
- `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| H1 | The P12 call is taken | `p12`'s stderr note names a P12 logits kernel (no fallback note); `dflt` prints none |
| H2 | Same candidates, same ids | (a) speed.json, 19 prompts, `INTARSIA_SPEC_TRACE=1`: proposed / accepted / passes **equal** `dflt` vs `p12` and the ids md5 equal. (b) g3.json md5 `off` = `dflt` = `p12` = the goldens (code `c69d1438...`, len4k `9e85ba11...`, len16k `89994b87...`) |
| H3 | The mechanism | median `head` phase a pass (unprofiled, `raw/phases.py`): `p12` < `dflt` (slot G: 1.93 ms) |
| H4 | Speed | decode ABBA x 4, speed.json, 45 s idle before each run: `raw/rule.py H/d p12 dflt` with `RULE_BAR=1.01`, plus the macmon power grouping (`raw/power.py`) |

**What follows:**
- H1-H3 pass and H4 GO: a flip PR is the user's call. The PR states the measured gain and the revert (`INTARSIA_DFLASH2_HEAD=bf16`).
- H4 OPT-IN or NO-GO with H1-H3 passing: the change stays opt-in and the doc says so.
- H1 or H2 fails: stop and find why before any speed claim.

## Slot H results (fleetq #12947, endor exclusive, binary md5 `4c03e08b...`, commit `031689231`)

**H1 (the P12 call is taken): pass.** `p12` prints `dflash: DFlash2 draft head: the target's P12 unembed (matmul_p12_logits_gemv_rows,
INTARSIA_DFLASH2_HEAD=p12)`; `dflt` prints no note.

**H2 (same candidates, same ids): pass.**
- (a) speed.json: `dflt` and `p12` both have passes 1195, proposed 4244, accepted 3030 (71.4 %). All 19 prompts' ids md5 are equal.
- (b) g3.json: md5 `off` = `dflt` = `p12` = the goldens (code `c69d1438...`, len4k `9e85ba11...`, len16k `89994b87...`). Counts 193 / 758 / 565 on both spec arms.
- Every ABBA run in H4 has the same counts.

**H3 (the mechanism): pass.** Median ms a pass over 1231 passes, unprofiled:

| arm | flush | block | head | selproj | dl | total |
|---|---:|---:|---:|---:|---:|---:|
| dflt (bf16 head) | 0.737 | 2.486 | **1.895** | 0.212 | 0.410 | 5.832 |
| p12 | 0.767 | 2.612 | **1.476** | 0.220 | 0.409 | 5.559 |

The head drops 0.42 ms (-22 %). That is 692 vs 922 MB at the same rate: 1.476 ms is 469 GB/s of P12 bytes, against 487 GB/s for the
bf16 head. The draft total drops 0.27 ms. The block phase went up 0.13 ms in this pair; the head does not run inside it, so that rise
is not this change, but this slot does not show its cause.

**H4 (speed): GO.** `rule.py H/d p12 dflt`, `RULE_BAR=1.01`:

| cell | ratio | 95 % CI |
|---|---:|---|
| short | 1.029 | 0.993-1.067 |
| 1k | 1.016 | 1.000-1.032 |
| prose | 1.019 | 1.010-1.029 |
| chat | 1.028 | 0.994-1.064 |
| 4k | 1.020 | 1.016-1.024 |
| 16k | 1.010 | 1.003-1.018 |

Geo mean 1.020 [1.008-1.033]. All six cells are faster by the point estimate, and no CI upper bound is below 0.99.

Power grouping (`raw/power.py`, macmon 250 ms):
- `dflt@r1` ran cooler: 1588 MHz, 42.9 W, 119.28 tok/s. That inflates its rep, 1.031.
- The other seven runs ran at 1617 MHz:
  - `dflt`: 122.71 / 122.72 / 123.03, mean 122.82.
  - `p12`: 124.38 / 124.66 / 124.98 / 125.12, mean 124.79.
- **Matched power: 1.016x.** Reps r2-r4 by the rule: 1.013 / 1.021 / 1.017.

**Reading:** a bitwise +1.6 % on Metal decode, all from the head phase, as predicted (0.4-0.5 ms of a ~28.7 ms pass). Memory is
unchanged: the P12 copy is already resident for the target's head.

**Revert / flip:** `INTARSIA_DFLASH2_HEAD=bf16` (or unset) is the current behaviour. Making `p12` the Metal default would be a one-line
change in `dflash2_head_p12` (unset on Metal = `p12`). That waits for the user's sign-off.
