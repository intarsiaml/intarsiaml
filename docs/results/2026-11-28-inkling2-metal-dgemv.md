# Inkling-2-Mini on Metal: the DFlash2 draft's GEMVs (lane ink-metal-dgemv, 2026-11-28)

Status: **closed, findings only (no code change).** Slot G (fleetq #12937): the draft on the ordinary kernel picks is 13 % slower
(NO-GO); the row-exact default stays. Slot K (fleetq #12940): the row-exact kernel already reads the draft's weights at 510-525 GB/s
at 8 rows; slot P's 132 GB/s for 6912x2304 was a per-dispatch-profile artifact. No GEMV lever left in the draft.

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

## Slot G results (fleetq #12937, endor exclusive, binary md5 `978535cb...`)

**G2 (ids): pass.** g3.json md5 `off` = `dflt` = `rk0` on all three prompts (code `c69d1438...`, len4k `9e85ba11...`, len16k
`89994b87...`); acceptance 74.5 % / 74.8 %.

**G1 (draft phases, ms a pass, median over ~1230 passes):**

| arm | flush | block | head | selproj | dl | total | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| dflt (row-exact) | 0.749 | **2.543** | **1.932** | 0.217 | 0.415 | **5.947** | 71.4 % |
| rk0 (ordinary picks) | 0.815 | 5.015 | 5.470 | 0.231 | 0.415 | 12.038 | 71.5 % |

**G0 (rk0 profile, 166 passes):** the ordinary picks send the draft's 8-row matmuls to `matmul_bf16_nax` (6912x2304 0.897 ms a pass,
288x2304 0.795, 1024x2304 0.836, 2304x6912 0.699) and the draft head to `logits_bf16 200058x2304` at **4.826 ms a pass** (14.2 % of
GPU time): the NAX GEMM tiles cost more than the row-exact GEMV at 8 rows, and the head is the bulk of the loss.

**G3 (decode ABBA x 4, speed.json, `RULE_BAR=1.01`): NO-GO.** rk0 / dflt:

| cell | ratio | 95 % CI |
|---|---:|---|
| short | 0.867 | 0.797-0.943 |
| 1k | 0.861 | 0.813-0.912 |
| prose | 0.858 | 0.827-0.891 |
| chat | 0.864 | 0.797-0.937 |
| 4k | 0.880 | 0.866-0.894 |
| 16k | 0.870 | 0.865-0.874 |

Geo mean 0.867 [0.828-0.908]; every cell's CI upper bound < 0.99. Overall tok/s: dflt 113.99 / 123.18 / 123.02 / 123.10, rk0 104.50 /
104.26 / 104.44 / 104.48 (dflt r1 ran cold). Ids identical across every process.

**What follows:** the row-exact default is right for the draft on Metal. The remaining lever is the row-exact kernel itself on the
short-K, wide-N shape (6912x2304 at 132 GB/s against 460 GB/s for 2304x6912). Slot K (`raw/slotK.sh`, the `gemv_g0` example with
the Inkling draft cases `ink_gu`, `ink_gup`, `ink_down`, `ink_q`, `ink_kv` at 7 and 8 rows) measures every candidate row kernel
per shape, multi-row forms checked bitwise against 8 serving 1-row launches first.

## Slot K results (fleetq #12940, endor exclusive, `gemv_g0` md5 `a09b564d...`, 20 interleaved rounds, cold weight pool)

All 80 bitwise checks (every multi-row form, 1 / 7 / 8 / 6 / 16 rows, against the serving 1-row launches): **IDENTICAL**.

| shape (8 rows) | serving 1-row x1 | mr8x1 (the default) | mr8x2 | mr8x4 | nax16 |
|---|---:|---:|---:|---:|---:|
| ink_gu 6912x2304 | 57.0 us, 559 GB/s | **60.7 us, 524 GB/s** | 61.5 | 135.6 | 90.4 |
| ink_gup pair 2x6912x2304 | 111.6 us, 571 GB/s | 120.2 us, 530 GB/s | 117.6 | 252.6 | 160.1 |
| ink_down 2304x6912 | 57.6 us | 62.2 us, 512 GB/s | 70.9 | 141.8 | 141.6 |
| ink_q 4096x2304 | 34.1 us | 37.5 us, 503 GB/s | 38.8 | 87.0 | 64.9 |
| ink_kv 1024x2304 | 10.1 us | 11.5 us, 409 GB/s | 12.3 | 34.9 | 40.0 |

(`serving 1-row x1` is one 1-row launch: the time to read the weights once. Per-row times for 7 rows are in `raw/K/g0.err`.)

**Reading:**

1. The 8-row row-exact GEMV costs 6-14 % over one weight read on every draft shape. That is close to the bandwidth floor. Slot P's
   0.242 ms a dispatch for 6912x2304 is 4x what the kernel takes alone, and the unprofiled draft block phase is 2.54 ms a pass in
   total (G1), less than the 2.42 ms the profile charged to gate and up alone. The per-dispatch profile inflated that row; it is
   not a kernel cost.
2. More output rows per threadgroup (`mr8x2`, `mr8x4`) do not help: equal at best (gate / up pair +2 %), up to 2.5x slower.
3. The fused pair for gate + up (one launch, the activation read once) saves 1 % against two `mr8x1` launches (120.2 vs 121.4 us).
   That is not worth a code change.
4. The NAX GEMM (`nax16`) is 1.5-4x slower than the row-exact GEMV at these shapes, which is G0's finding in slot G seen from the
   kernel side.

**What follows:** no draft-GEMV lane. The per-dispatch profile (`prof4.json`) charges some kernels about 4x their isolated cost,
so its shares cannot rank levers: cross-check them against the unprofiled phase totals (`INTARSIA_SPEC_TRACE=1`) and isolated kernel
benches before opening a lane. (The target's MoE experts, the profile's top row, already stream at 480-550 GB/s in isolation:
`2026-11-21-inkling2-metal-moe.md`.) A lever that cuts bytes on a kernel already at bandwidth is the kind left: the draft head reads
the bf16 unembed (1.71 ms a pass at 539 GB/s) while the target's verify head reads its lossless P12 copy, 25 % fewer bytes and
bitwise the same logits (lane ink-metal-p12head).
