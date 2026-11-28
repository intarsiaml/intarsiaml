# Inkling-2-Mini on Metal: the P12 draft head and the device top-k as defaults (lane ink-metal-flip3, 2026-11-28)

Status: **DRAFT; the flip waits on the user's sign-off.** The branch stacks on #560 (`ink/metal-p12head`). Confirm slot F3 (fleetq #12974)
passes F1-F3: GO, Metal decode **1.025x** in matched power, bitwise.

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

## Slot F3 results (fleetq #12974, endor exclusive, binary md5 `6a4ba6be...`, commit `167d227de`)

Tests (fleetq #12973): core `dflash` 9/9, models `inkling2_dflash2` 14/14.

**F1 (defaults resolve): pass.** `new` prints `dflash: candidates by the Metal device top-k (default; INTARSIA_DFLASH_DEVICE_TOPK=0 takes
the host's)` and `dflash: DFlash2 draft head: the target's P12 unembed (matmul_p12_logits_gemv_rows; ...)`. `rev` prints neither.

Draft phases, ms a pass, median over 1231 passes, unprofiled:

| arm | flush | block | head | selproj | dl | select | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev (old default) | 0.747 | 2.490 | 1.908 | 0.217 | 0.410 | 0.000 | 5.869 |
| new | 0.772 | 2.608 | **1.478** | 0.221 | **0.279** | 0.017 | **5.405** |

**F2 (bitwise): pass.**
- speed.json: 1195 / 4244 / 3030 on both arms, and all 19 ids md5 equal.
- g3.json md5: `off` = `rev` = `new` = the goldens (code `c69d1438...`, len4k `9e85ba11...`, len16k `89994b87...`).
- Every ABBA run in F3 has the same counts, and ids are identical across all eight processes.

**F3 (speed): GO.** geo 1.030 [1.007-1.054]; every cell faster by the point estimate, none below 0.99:

| cell | ratio | 95 % CI |
|---|---:|---|
| short | 1.040 | 0.997-1.085 |
| 1k | 1.027 | 0.982-1.074 |
| prose | 1.025 | 1.014-1.037 |
| chat | 1.039 | 0.999-1.081 |
| 4k | 1.029 | 1.024-1.034 |
| 16k | 1.021 | 1.013-1.029 |

Power grouping: the script logged macmon to `.scratch/macmon-H.jsonl` (a name left over from slot H's script; the data is slot F3's).
- `rev@r1` ran cooler: 1566 MHz, 42.8 W, 118.52 tok/s.
- The other seven runs ran at 1617 MHz:
  - `rev`: 122.65 / 122.87 / 122.90, mean 122.81.
  - `new`: 126.07 / 125.43 / 126.00 / 126.05, mean 125.89.
- **Matched power: 1.025x.** Slot C measured the same pair through env vars and got 1.026x.

**Reverts:** `INTARSIA_DFLASH2_HEAD=bf16` (draft head) and `INTARSIA_DFLASH_DEVICE_TOPK=0` (host top-k), separately or together.
