#!/usr/bin/env bash
# Lane srv16, GPU window 1 (one fleetq slot): (1) nsys + host profile of main under the 16-stream ~1k-prompt burst (wall-time
# attribution, timeline.py), (2) the packed-prefill gate on dev (batch_cuda_real, ignored test) and served-token identity base vs
# dev + INTARSIA_PACK_PREFILL=1 (equal.py + ident_long.py), (3) one exploratory localeval round (base, dev + packing, base +
# packing (not bitwise on main)), (4) nsys of dev + packing. A stage that would start past 38 min is skipped.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "srv16: W1 attribution + pack gate" -- scripts/srv16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/srv16/docs/results/2026-10-28-srv16-serve-gap/raw
cd "$H/../.."
T0=$SECONDS
go() { if (( SECONDS - T0 > 38 * 60 )); then echo "SKIP (time): $*"; return 1; fi; date +%T; echo "== $*"; }
go prof base && N=16 "$H/prof.sh" base base-b16 INTARSIA_HOST_PROFILE=1
go packed gate && "$H/batch-real.sh" packed | tee "$RAW/batch-real-packed.txt"
go identity && { "$H/ident.sh" base:base devpack:dev:INTARSIA_PACK_PREFILL=1; echo "identity exit $?"; }
go localeval && "$H/le-window.sh" w1 1 base:base::--max-batch_16 devpack:dev:INTARSIA_PACK_PREFILL=1:--max-batch_16 basepack:base:INTARSIA_PACK_PREFILL=1:--max-batch_16
go prof devpack && N=16 "$H/prof.sh" dev devpack-b16 INTARSIA_HOST_PROFILE=1 INTARSIA_PACK_PREFILL=1
date +%T
