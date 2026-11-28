# Inkling-2-Mini on Metal: a quantised DFlash2 draft head copy (mxfp4 / fp8) — probe, NO-GO (lane ink-metal-dhead4, 2026-11-28)

Status: **closed after the probe (slot Q, fleetq #12967); no code change.** Metal keeps refusing `INTARSIA_DFLASH2_HEAD=mxfp4|fp8`.
The fastest Metal draft head is the target's own P12 unembed (`INTARSIA_DFLASH2_HEAD=p12`, #560).

## Question

On HIP the DFlash2 draft scores candidates with its own mxfp4 copy of the unembed: 245 MB against P12's 692 MB. On Metal that copy
was refused because it had never been measured. The question was whether it is faster than the P12 head. The draft's numerics affect
only acceptance; the ids stay the verify's.

Gate, set before the probe: go on to a full lane only if the draft head phase lands clearly under ~1.2 ms a pass. The P12 head takes
1.47 ms. The advisor's prior: at 7 rows the head does 3.23 G FMA a pass, and the Metal M-row MXFP4 GEMV runs at 2.75-2.94 TFMA/s
(`2026-11-06-metal-dflash2.md`). That makes it compute-bound at about 1.1-1.2 ms, not the ~0.5 ms the byte count suggests.

## Probe (slot Q, endor exclusive, binary md5 `db84fd66...`)

The probe binary is main + #560 with the Metal refusal lifted. One `INTARSIA_SPEC_TRACE=1` pass set per arm on speed.json (19 prompts,
256 tokens), all arms with device top-k on, plus g3.json ids.

| arm | kernel | head ms a pass | draft total ms | passes / proposed / accepted | tokens a pass |
|---|---|---:|---:|---|---:|
| p12 (#560) | `matmul_p12_logits_gemv_rows` | **1.471** | 5.303 | 1195 / 4244 / 3030 | 3.54 |
| mxfp4 copy | `matmul_mxfp4_gemv_rows` (exact_rows) | 1.524 | 5.791 | 1204 / 4259 / 3014 | 3.50 |
| fp8 copy | `matmul_fp8_gemv_rows` (exact_rows) | 1.684 | 5.905 | 1196 / 4232 / 3025 | 3.53 |

- Both copies take an M-row kernel; neither falls back to one-row launches.
- Neither copy is faster than P12. The mxfp4 head is 1.524 ms, about 4.7 TFMA/s equivalent, so it is not memory-bound.
- mxfp4 also loses a little acceptance: 3.50 against 3.54 tokens a pass.
- Ids: all 19 speed.json prompts equal across the three arms, and g3.json md5 equals the goldens on both copies. The verify keeps
  outputs exact, as expected.
- The traced decode totals are single runs, not a matched-power A/B: p12 120.06, mx 108.86, f8 111.34 tok/s.

**Reading:** at the draft's 7 rows, the quantised GEMVs pay their dequantisation compute on every row. P12's 12-bit unpack plus the
bf16 path is cheaper than mxfp4's e2m1 path. Fewer bytes do not help once a kernel is compute-bound. NO-GO without a full A/B; the
refusal stays.
