# Inkling-2-Mini on Metal: the P12 draft head and the device top-k as defaults (lane ink-metal-flip3, 2026-11-28)

Status: **DRAFT; the flip waits on the user's sign-off.** The branch stacks on #560 (`ink/metal-p12head`). Confirm slot F3 not run yet.

## Change (Metal only; HIP and CPU unchanged)

- `INTARSIA_DFLASH2_HEAD` unset on Metal = `p12`. The DFlash2 draft head reads the target's lossless P12 unembed, which is bitwise
  the bf16 head. `bf16` / `0` / `off` keep the bf16 head (`head_p12_resolve`, `crates/intarsia-core/src/model/dflash.rs`).
- `INTARSIA_DFLASH_DEVICE_TOPK` unset on Metal = on. Candidates come from the Metal device top-k, bit-identical to the host's. `0`
  takes the host's (`device_topk_resolve`, `crates/intarsia-core/src/model/dflash2.rs`).
- Unit test `dflash2` resolve asserts both; tuning.md rows updated.

Evidence so far: slot H #12947 (P12 head alone, 1.016x matched power, bitwise) and slot C #12955 (both together: 0 differing rows
on 1231 checked passes, g3.json = goldens, **1.026x** matched power), `2026-11-28-inkling2-metal-p12head.md`.

## Confirm checks (pre-registered before slot F3)

One binary (`intarsia-f3`), endor `--exclusive`, `raw/slotF3.sh`. Arms:
- `new` = no env (the new default).
- `rev` = `INTARSIA_DFLASH2_HEAD=bf16 INTARSIA_DFLASH_DEVICE_TOPK=0` (the old default).
- `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| F1 | Defaults resolve | `new` prints the P12 head note and the device top-k note; `rev` prints neither |
| F2 | Bitwise | speed.json passes / proposed / accepted and all 19 ids md5 **equal** `rev` vs `new`; g3.json md5 `off` = `rev` = `new` = the goldens |
| F3 | Speed | decode ABBA x 4 with 45 s idle gaps, `rule.py F3/d new rev` with `RULE_BAR=1.01`, plus the macmon matched-power grouping |

Tests in the build (`raw/build.sh`): core `dflash` and models `inkling2_dflash2`.
