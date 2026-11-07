# Adaptive MTP draft depth on the R9700: 27B tp1 (2026-11-07)

**Result: adaptive depth 2-4 (`INTARSIA_SPEC_K_MIN=2 --spec-k 4`) is the fastest of the three arms, greedy and sampled.
Greedy outputs are byte-identical to fixed k2 and k4.**

| requests | adaptive 2-4 vs fixed k2 | adaptive 2-4 vs fixed k4 | fixed k4 vs fixed k2 |
|---|---|---|---|
| greedy (temperature 0, the default path) | **+7.0%** | +1.6% | +5.3% |
| sampled, `INTARSIA_SPEC_SAMPLED=reject`, temperature 1.0 | **+5.8%** | +1.0% | +4.7% |

Geomean decode tok/s over 7 categories, 2 rounds per window with the arm order reversed in round 2. Fixed k4 is +15-21% on
code and -5% on prose against k2. Adaptive keeps prose at k2's speed (-0.4% to 0%), gets most of k4's code gain (code +11-18%
against k2, 1-4% under k4), and beats both on chat (+3-5% against k2). It gives up some code speed because every request
starts at depth 2 and climbs: 512-token outputs spend part of their passes below 4.

## What it is

Ported from llamAmpere's `draft-mtp-adaptive` (`common/speculative-adaptive.h`, `common_speculative_adaptive`), same
thresholds. `INTARSIA_SPEC_K_MIN=n` with `n` below `--spec-k` turns it on for MTP drafts (`--spec mtp`; not DFlash, not
prompt lookup). Each request starts at depth `n`. The depth climbs one step after a streak of fully accepted passes (2, 4,
6, 5, 4, 3 passes at depth 1-6: cheap at the ends, costly around 3, where prose acceptance falls off) and drops one step
after `max(20, 5 x depth)` drafts rejected since the last full accept; it stays in `[n, --spec-k]`. With mb-MTP on, the
controller counts only the drafts exact acceptance would have kept (llamAmpere's MBFIX), so mb does not move the depth.
Code: `spec::Depth`, `Seq::draft_k` and `Seq::account_verify` in `engine.rs`. Unit test `depth_climbs_and_falls`.

## Setup

tatooine, one R9700 (card 0), Qwen3.8-27B Quark AWQ MXFP4, `serve --backend hip --spec mtp --max-batch 1`, binary from
`spec-adaptive-k` 9b56b92fe (fleetq #9027). Window `scripts/dt1/cat-window.sh`, 7 category files, 512 forced tokens per
request, 2 rounds (order a b c, then c b a). Arms: `mtp2` = `--spec-k 2`, `mtp4` = `--spec-k 4`, `mtp4-on` = `--spec-k 4`
plus `INTARSIA_SPEC_K_MIN=2`. Tables: `tools/mb/table.py <window> <arm A> <arm B>`.

## Greedy (window adk2)

fleetq #9029, radiance down 11:08:28-11:47:14.

Adaptive 2-4 (`mtp4-on`) against fixed k2 (`mtp2`):

| category | mtp2 tok/s | mtp4-on tok/s | gain | accepted per pass | draft acceptance | requests |
|---|---|---|---|---|---|---|
| ho-chat | 64.1 | 67.6 | +5.4% | 1.25 / 1.56 | 62.5% / 54.7% | 8 / 8 |
| ho-code | 79.8 | 94.3 | +18.2% | 1.79 / 2.81 | 89.6% / 77.7% | 4 / 4 |
| ho-prose | 59.1 | 58.9 | -0.2% | 1.08 / 1.13 | 54.1% / 49.5% | 8 / 8 |
| ho-recall | 69.6 | 73.2 | +5.0% | 1.44 / 1.86 | 72.3% / 58.4% | 4 / 4 |
| rcp-code | 78.1 | 89.5 | +14.6% | 1.73 / 2.64 | 86.6% / 72.9% | 16 / 16 |
| rcp-prose | 60.0 | 59.7 | -0.4% | 1.11 / 1.17 | 55.7% / 50.2% | 16 / 16 |
| rcp-recall | 70.2 | 75.8 | +7.9% | 1.46 / 1.90 | 73.3% / 63.6% | 16 / 16 |

geomean gain +7.0%

Adaptive 2-4 against fixed k4 (`mtp4`):

| category | mtp4 tok/s | mtp4-on tok/s | gain | accepted per pass | draft acceptance | requests |
|---|---|---|---|---|---|---|
| ho-chat | 65.5 | 67.6 | +3.3% | 1.74 / 1.56 | 43.6% / 54.7% | 8 / 8 |
| ho-code | 96.3 | 94.3 | -2.1% | 3.02 / 2.81 | 75.9% / 77.7% | 4 / 4 |
| ho-prose | 56.0 | 58.9 | +5.3% | 1.35 / 1.13 | 33.7% / 49.5% | 8 / 8 |
| ho-recall | 72.5 | 73.2 | +0.8% | 2.04 / 1.86 | 51.2% / 58.4% | 4 / 4 |
| rcp-code | 90.3 | 89.5 | -0.9% | 2.80 / 2.64 | 70.3% / 72.9% | 16 / 16 |
| rcp-prose | 56.9 | 59.7 | +5.0% | 1.39 / 1.17 | 34.7% / 50.2% | 16 / 16 |
| rcp-recall | 75.7 | 75.8 | +0.1% | 2.16 / 1.90 | 54.3% / 63.6% | 16 / 16 |

geomean gain +1.6%

**Outputs:** all 72 request outputs (36 per round) are byte-identical between `mtp2`, `mtp4` and `mtp4-on`, and between the
two rounds of each arm.

## Sampled, reject mode (window adk1)

fleetq #9028, radiance down 10:27:43-11:07:36, `TEMP=1.0`, all arms `INTARSIA_SPEC_SAMPLED=reject`.

Adaptive 2-4 against fixed k2:

| category | mtp2-on tok/s | mtp4-on2 tok/s | gain | accepted per pass | draft acceptance | requests |
|---|---|---|---|---|---|---|
| ho-chat | 63.2 | 65.4 | +3.4% | 1.31 / 1.58 | 65.6% / 56.2% | 8 / 8 |
| ho-code | 75.0 | 83.3 | +11.1% | 1.75 / 2.51 | 87.5% / 72.3% | 4 / 4 |
| ho-prose | 56.9 | 56.9 | -0.0% | 1.09 / 1.15 | 54.7% / 50.3% | 8 / 8 |
| ho-recall | 63.8 | 67.1 | +5.1% | 1.34 / 1.69 | 67.0% / 58.5% | 4 / 4 |
| rcp-code | 73.3 | 83.9 | +14.5% | 1.69 / 2.59 | 84.4% / 72.9% | 16 / 16 |
| rcp-prose | 57.6 | 57.6 | -0.1% | 1.11 / 1.20 | 55.8% / 50.2% | 16 / 16 |
| rcp-recall | 67.0 | 71.9 | +7.3% | 1.46 / 1.92 | 73.3% / 62.7% | 16 / 16 |

geomean gain +5.8%

Adaptive 2-4 against fixed k4:

| category | mtp4-on tok/s | mtp4-on2 tok/s | gain | accepted per pass | draft acceptance | requests |
|---|---|---|---|---|---|---|
| ho-chat | 63.0 | 65.4 | +3.8% | 1.76 / 1.58 | 44.0% / 56.2% | 8 / 8 |
| ho-code | 86.4 | 83.3 | -3.6% | 2.84 / 2.51 | 71.3% / 72.3% | 4 / 4 |
| ho-prose | 54.1 | 56.9 | +5.1% | 1.38 / 1.15 | 34.6% / 50.3% | 8 / 8 |
| ho-recall | 67.4 | 67.1 | -0.4% | 1.98 / 1.69 | 49.7% / 58.5% | 4 / 4 |
| rcp-code | 86.1 | 83.9 | -2.5% | 2.82 / 2.59 | 70.8% / 72.9% | 16 / 16 |
| rcp-prose | 54.9 | 57.6 | +4.9% | 1.41 / 1.20 | 35.5% / 50.2% | 16 / 16 |
| rcp-recall | 71.8 | 71.9 | +0.1% | 2.18 / 1.92 | 54.6% / 62.7% | 16 / 16 |

geomean gain +1.0%

## Tensor parallel, greedy (window adk3)

27B at tp2 over both R9700s (the recipe's MTP default applies only here on HIP), greedy, 2 rounds, arms `--spec-k 2`, `--spec-k 4`,
`--spec-k 4` with `INTARSIA_SPEC_K_MIN=2`. Per-round medians agree within 1 %.

| category | k2 tok/s | k4 tok/s | adaptive 2..4 tok/s | adaptive vs k2 | adaptive vs k4 |
|---|---|---|---|---|---|
| ho-chat | 100.7 | 100.5 | 102.1 | +1.4% | +1.6% |
| ho-code | 121.3 | 149.0 | 148.3 | +22.2% | -0.5% |
| ho-prose | 89.7 | 83.6 | 89.3 | -0.5% | +6.9% |
| ho-recall | 106.3 | 111.4 | 111.1 | +4.5% | -0.4% |
| rcp-code | 119.0 | 139.7 | 137.2 | +15.3% | -1.8% |
| rcp-prose | 90.9 | 85.3 | 91.0 | +0.1% | +6.7% |
| rcp-recall | 108.0 | 117.5 | 115.2 | +6.7% | -1.9% |
| geomean | | | | **+6.8%** | **+1.5%** |

The tp1 picture holds at tp2: adaptive keeps k4's code gain within 2 % and k2's prose speed. On this basis the recipe's HIP tp default
became adaptive 2..4 (`[tuning.hip] mtp_default_tp_k = 4`, `mtp_default_k_min = 2`; user decision 2026-11-07 "Yes, flip the default").

**Served default check (window adk4, branch of PR #341):** `serve --tp 2` with no `--spec` flags logs "--spec mtp default drafts
2..4 tokens per pass" and runs +6.6 % geomean over `--spec-k 2` (code +15 to +21 %, prose -0.5 to -0.7 %, 1 round); all 36 greedy
outputs are byte-identical to `--spec-k 2`.

## Limits

- One model (27B), tp1, batch 1, 512-token outputs at short context. Longer outputs should favour adaptive more (less of
  the request spent climbing); not measured.
- Only the range 2-4. Not tried: other floors or ceilings, starting at the ceiling, CUDA, Metal, the batching scheduler
  (the depth is per sequence, so it applies there too, untested).
- Not combined with mb-MTP in a window (mb-MTP alone: docs/results/2026-11-06-mb-mtp-hip.md).

Raw: `docs/results/2026-11-06-27b-dflash-hip-tp1/raw/adk1`, `.../raw/adk2`, `.../raw/adk3`, `.../raw/adk4`; localeval runs `results/*dt1-adk{1,2,3}-*`.
