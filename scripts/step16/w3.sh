#!/usr/bin/env bash
# Lane step16, GPU window 3 (one fleetq slot on spark1, quiet on GPU and CPU): W2's nsys profiles again (W2's burst.py read a stale corpus
# path, so its profiles held no decode steps): base and dev under a 16- and an 8-stream burst (prof.sh, per-kernel tables). A stage
# that would start past 38 min is skipped.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "step16: W3 nsys" -- scripts/step16/w3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/step16/docs/results/2026-10-29-step16-decode-step/raw; mkdir -p $RAW
cd "$H/../.."
T0=$SECONDS
go() { if (( SECONDS - T0 > 38 * 60 )); then echo "SKIP (time): $*"; return 1; fi; date +%T; echo "== $*"; }
go prof base-b16 && N=16 "$H/prof.sh" base base-b16
go prof dev-b16 && N=16 "$H/prof.sh" dev dev-b16
go prof dev-b8 && N=8 "$H/prof.sh" dev dev-b8
go prof base-b8 && N=8 "$H/prof.sh" base base-b8
date +%T
