#!/usr/bin/env bash
# Lane srv16, GPU window 2 (one fleetq slot), dev3 = packed prefill on by default on CUDA, a pass up to the macro workspace:
# exactness gates (batch_cuda_real packed + batched, rows_exact, served identity base vs dev3), the 3-round interleaved A/B
# base vs dev3 at 1/4/8/16 streams, then nsys of dev3 for the attribution table. A stage past 40 min is skipped.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "srv16: W2 gates + A/B" -- scripts/srv16/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/srv16/docs/results/2026-10-28-srv16-serve-gap/raw
cd "$H/../.."
T0=$SECONDS
go() { if (( SECONDS - T0 > 40 * 60 )); then echo "SKIP (time): $*"; return 1; fi; date +%T; echo "== $*"; }
go packed gate && "$H/batch-real.sh" packed | tee "$RAW/w2-batch-real-packed.txt"
go batched gate && "$H/batch-real.sh" run | tee "$RAW/w2-batch-real.txt"
go rows_exact && "$H/test.sh" run --test rows_exact | tee "$RAW/w2-rows-exact.txt"
go identity && { IDENT_OUT=ident-w2 "$H/ident.sh" base:base dev3:dev3; echo "identity exit $?"; }
go A/B && "$H/le-window.sh" w2 3 base:base::--max-batch_16 dev3:dev3::--max-batch_16
go prof dev3 && N=16 "$H/prof.sh" dev3 dev3-b16 INTARSIA_HOST_PROFILE=1
date +%T
