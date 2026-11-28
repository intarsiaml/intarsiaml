# Inkling-2-Mini on Metal: the DFlash2 draft head on the target's P12 unembed (lane ink-metal-p12head, 2026-11-28)

Status: **checks pre-registered, slot H not run yet.** Opt-in (`INTARSIA_DFLASH2_HEAD=p12`); no default change.

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
