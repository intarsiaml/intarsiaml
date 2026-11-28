# Inkling-2 on Metal: the prefill KV prologue split, re-measured on today's main

Lane ink-prologue2, 2026-11-27. It reopens a lever from lane ink-prefill (`docs/results/2026-11-21-inkling2-prefill.md`).

## Why reopen it

- **The kernel:** `INTARSIA_INK_PROLOGUE_SPLIT=1` runs InkKvPrologue for prefill chunks as two kernels: the prologue without the rel
  product, plus the rel product per 8-row block (`kernels/metal/inkling2_prologue_split.metal`). It is bitwise the single
  `ink_kv_prologue_bf16`. In the 11-22 trace it cut the prologue from 48.0 to 24.1 ms per 2048-row chunk (GPU time per chunk -3.5 %).
- **The 11-22 verdict:** it stayed opt-in, final under brief 19. 1k generate timing passed (1.007x). 4k failed: median 0.956x, with
  one "on" run at 1700 tok/s against 1890-2010 for the other five runs. 8k (+3.6 %) and 16k (+4 to 8 %) were faster.
- **What changed since:** on 11-26 we found that endor's Metal speed depends on its power history (GPU burst). Back-to-back runs
  without idle gaps can be fast or slow for that reason alone. The 11-22 pairs had no idle gaps. So the 4k verdict may be the box's
  state, not the kernel's. This lane re-measures with 45 s idle gaps before every timed run and macmon logged, the method used since.
- **The branch:** the 11-22 commits `c8bbcad89` (kernel, opt-in, bitwise test) and `e7f943dc7` (launch counter) cherry-picked onto
  main 351657cc1 without conflicts. Since 11-22, `InkP` in `inkling2_attn.metal` only gained fields at its end (`869be5b87`, the
  row table), so the split shader's mirror of its prefix (`PsP`) keeps the same offsets. P1 checks that on the GPU.

## Gates (pre-registered 2026-11-27 ~20:40 EST, before the first endor slot)

One binary (this branch, the split still opt-in), one endor `--exclusive` fleetq slot, script `raw/slotP.sh`. Arms: `off` = default,
`on` = `INTARSIA_INK_PROLOGUE_SPLIT=1`. Prompts `raw/pf3.json`: p1k (1000 ids), len4k (4108), len16k (16455), one process per run.

| # | Check | Rule |
|---|---|---|
| P1 | Bitwise kernel | `ink_prologue_split` test passes on the GPU (split = `ink_kv_prologue_bf16` bit for bit, 17-2048 rows, split path ran) |
| P2 | Greedy ids | `--spec off`, 64 ids per prompt: md5 `on` = `off` on all three; `on`'s trace shows `prologue split`, `off`'s does not |
| P3 | Prefill speed | 12 runs, ABBA x 3, 45 s idle before each, `generate --max-tokens 1`. Median on / off: **p1k >= 0.98, len4k >= 1.00, len16k >= 1.00** |

Brief 19's thresholds are kept: 1k >= 0.98, 4k >= 1.00, with 16k >= 1.00 added.

**Decision rule:**
- P1-P3 all pass: PR with the split opt-in (this branch). Then ask the user to flip it default-on, in its own commit, with
  `INTARSIA_INK_PROLOGUE_SPLIT=0` as the revert. It is bitwise, but a default change needs their sign-off.
- P1 or P2 fails: the kernel no longer matches today's prologue. Report it and merge nothing.
- P3 fails: the split stays off main (not merged). Report the numbers.

Not measured: decode (the split acts only on prefill chunks of more than 16 rows), and serve-path TTFT.

## Results

(pending)
