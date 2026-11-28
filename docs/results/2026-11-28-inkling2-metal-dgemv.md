# Inkling-2-Mini on Metal: the DFlash2 draft's GEMVs (lane ink-metal-dgemv, 2026-11-28)

Status: **checks pre-registered, slot G not run yet.**

## Why

Slot P of lane ink-metal-dtopk (`2026-11-28-inkling2-metal-dtopk.md`, a per-dispatch GPU profile of k7 + cut passes) found the
draft's MLP gate and up, `matmul_bf16_gemv_rows 6912x2304` at 8 rows (10 dispatches a pass), at **2.42 ms a pass: 0.242 ms each
for 31.9 MB, 132 GB/s**. The down projection `2304x6912` (same bytes, 8 rows) runs at 460 GB/s, `2304x4096` at 430 GB/s.

The draft runs its block pass and head on the row-exact `_rows` kernels because `INTARSIA_DFLASH_ROWS_KERNELS` (on by default for a
DFlash2 draft, `=0` off) sets `exact_rows` around them. That default was a speed choice measured on the 27B (lane dcp,
`2026-10-27-dcp-dflash-cost.md`), never on Inkling or Metal. The draft does not decide the output (greedy verification keeps the
target's tokens), so any kernel there keeps ids exact; only acceptance (candidate rounding) and speed can move.

## Checks (pre-registered before slot G; no code change)

One binary (`intarsia-dg` = lane ink-metal-dtopk's `intarsia-dt`, md5 `978535cb...`: main + the opt-in device top-k, which no arm
turns on), endor `--exclusive`, `raw/slotG.sh`. Arms: `dflt` = the Metal default; `rk0` = `INTARSIA_DFLASH_ROWS_KERNELS=0`;
`off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| G0 | Per-dispatch profile of `rk0` (prof4.json, 128 tokens) | reported: the draft's matmul kernels and ms a pass under the ordinary picks, against slot P's `dflt` table |
| G1 | Draft phases (`INTARSIA_SPEC_TRACE=1`, speed.json) | reported: median block / head / total a pass and acceptance, `dflt` vs `rk0` |
| G2 | Greedy ids | g3.json md5 `off` = `dflt` = `rk0` = the goldens |
| G3 | Decode ABBA x 4, speed.json | `raw/rule.py rk0 dflt` (`RULE_BAR=1.01`) plus the macmon power grouping (`raw/power.py`) |

**What follows:** if G3 shows `rk0` >= 1.01 in matched power with ids exact, the draft's kernel choice is worth a per-shape rule
(row-exact where it is at bandwidth, the ordinary pick where it is not); that is a code lane with its own gates, and any default
change is the user's call. If `rk0` is slower, the next step is a faster row-exact kernel for short-K, wide-N shapes.
